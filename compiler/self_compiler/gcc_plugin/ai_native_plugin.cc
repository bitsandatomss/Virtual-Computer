/* GCC AI-Native policy/telemetry plugin.

   This file is intentionally GPL-compatible because GCC refuses to load
   plugins that do not make that declaration.  The plugin runs inside cc1,
   after GIMPLE SSA construction, and exposes two compiler-native surfaces:

   1. structured per-function IR telemetry for an offline learned advisor;
   2. a conservative policy hook that may disable named optimization passes,
      globally or conditionally on the IR features of the current function.

   It never enables a pass whose GCC prerequisites may not hold and never
   invokes a network model from inside the compiler process.

   Policy file grammar (one rule per line, '#' starts a comment):

     disable_pass=<pass>
     disable_pass=<pass> if <feature><op><value>[&&<feature><op><value>]...

   <feature> is one of the telemetry feature names below, <op> is one of
   <= >= == != < >, and <value> is an integer.  Conditional rules are
   evaluated against the features captured after SSA construction, so a
   conditional rule can only influence passes that run later in the same
   function; earlier passes and functions without captured features keep
   GCC's default decision.  Malformed syntax is rejected at load time so a
   broken policy can never silently degrade to "no policy applied".  */

#include "gcc-plugin.h"
#include "plugin-version.h"

#include "context.h"
#include "tree.h"
#include "basic-block.h"
#include "cfg.h"
#include "dominance.h"
#include "function.h"
#include "gimple.h"
#include "gimple-iterator.h"
#include "input.h"
#include "tree-pass.h"
#include "diagnostic-core.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

int plugin_is_GPL_compatible;

namespace {

const char *plugin_name = "ai_native";
std::string output_path;
std::ofstream output;

/* Audit verbosity for conditional rules: gates = emit only decisions that
   disabled a pass; all = also emit gate-eval events for rules evaluated
   against a function whose predicates did not match.  The second mode
   provides the evaluation denominator needed to reason about rule
   coverage and exposure offline.  */
bool audit_all_evaluations = false;

/* Opt-in export of raw CFG connectivity per function.  Scalar structural
   features are always emitted; full edge lists are off by default because
   they dominate telemetry volume on large translation units.  */
bool cfg_export = false;

/* Per-function IR feature snapshot captured right after SSA construction.
   The field set is the versioned gcc-ai.telemetry.v3 schema: scalar totals
   plus structural topology summaries derived from dominance information
   (back edges, natural-loop nesting depth, dominator-tree height, McCabe
   cyclomatic complexity over the exported counters, and maximum branch
   fan-out).  Raw CFG connectivity is available behind cfg_export.  */
struct feature_snapshot
{
  unsigned long long basic_blocks = 0;
  unsigned long long gimple_statements = 0;
  unsigned long long phi_nodes = 0;
  unsigned long long calls = 0;
  unsigned long long branches = 0;
  unsigned long long edges = 0;
  unsigned long long memory_reads = 0;
  unsigned long long memory_writes = 0;
  unsigned long long float_ops = 0;
  unsigned long long back_edges = 0;
  unsigned long long max_loop_depth = 0;
  unsigned long long dominator_height = 0;
  unsigned long long cyclomatic_complexity = 0;
  unsigned long long max_out_degree = 0;
};

std::map<unsigned int, feature_snapshot> features_by_uid;

enum comparison_operator
{
  OP_LT,
  OP_LE,
  OP_EQ,
  OP_NE,
  OP_GE,
  OP_GT
};

struct predicate
{
  std::string feature;
  comparison_operator op;
  long long value;
};

struct policy_rule
{
  std::string pass_name;
  int instance = -1; /* -1 matches every instance of the pass.  */
  bool global = true;
  std::vector<predicate> predicates;
};

std::vector<policy_rule> rules;

std::string
json_escape (const char *text)
{
  std::ostringstream escaped;
  if (!text)
    return "";
  for (const unsigned char ch : std::string (text))
    {
      switch (ch)
        {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
          if (ch < 0x20)
            {
              const char hex[] = "0123456789abcdef";
              escaped << "\\u00" << hex[(ch >> 4) & 0xf] << hex[ch & 0xf];
            }
          else
            escaped << static_cast<char> (ch);
        }
    }
  return escaped.str ();
}

const char *
decl_name (tree declaration)
{
  if (!declaration || !DECL_NAME (declaration))
    return "<anonymous>";
  return IDENTIFIER_POINTER (DECL_NAME (declaration));
}

void
write_event (const std::string &event)
{
  if (output.is_open ())
    {
      output << event << '\n';
      output.flush ();
    }
}

/* Raw CFG connectivity for one function: successor triples of
   [source-index, dest-index, raw EDGE_* flag bitmask].  Emitted only when
   the cfg=export plugin argument is present because volume scales with
   graph size rather than program count.  */
std::string
cfg_event (function *fun)
{
  std::ostringstream event;
  event << "{\"schema\":\"gcc-ai.telemetry.v3\","
        << "\"event\":\"function-cfg\","
        << "\"function\":\"" << json_escape (decl_name (fun->decl)) << "\","
        << "\"succ\":[";
  bool first_edge = true;
  basic_block block;
  FOR_ALL_BB_FN (block, fun)
    {
      edge e;
      edge_iterator ei;
      FOR_EACH_EDGE (e, ei, block->succs)
        {
          if (!first_edge)
            event << ',';
          first_edge = false;
          event << '[' << e->src->index << ',' << e->dest->index << ','
                << static_cast<unsigned int> (e->flags) << ']';
        }
    }
  event << "]}";
  return event.str ();
}

/* Structural CFG summaries derived from dominance information, which is
   computed and maintained from SSA construction onward.  Back edges use
   the classic definition (an edge whose target dominates its source,
   ignoring abnormal/EH edges), natural loops give per-block nesting
   depth, and the dominator-tree height follows immediate-dominator
   links.  Loop processing is bounded so pathological control flow cannot
   stall compilation.  */
void
analyze_structure (function *fun, feature_snapshot *snapshot)
{
  calculate_dominance_info (CDI_DOMINATORS);

  const int total_blocks = last_basic_block_for_fn (fun);
  std::vector<int> loop_depth (total_blocks, 0);
  unsigned long long processed_loops = 0;

  basic_block block;
  FOR_ALL_BB_FN (block, fun)
    {
      edge e;
      edge_iterator ei;
      FOR_EACH_EDGE (e, ei, block->succs)
        {
          if ((e->flags & (EDGE_ABNORMAL | EDGE_EH)) != 0 || e->dest == NULL)
            continue;
          if (!dominated_by_p (CDI_DOMINATORS, e->src, e->dest))
            continue;
          ++snapshot->back_edges;
          if (processed_loops >= 256)
            continue;
          ++processed_loops;

          /* Natural loop of back edge src->dest: dest plus every block
             that reaches src without passing through dest.  */
          std::vector<char> member (total_blocks, 0);
          member[e->dest->index] = 1;
          std::vector<basic_block> pending;
          pending.push_back (e->src);
          while (!pending.empty ())
            {
              basic_block node = pending.back ();
              pending.pop_back ();
              if (member[node->index])
                continue;
              member[node->index] = 1;
              edge pe;
              edge_iterator pei;
              FOR_EACH_EDGE (pe, pei, node->preds)
                pending.push_back (pe->src);
            }
          for (int index = 0; index < total_blocks; ++index)
            if (member[index])
              ++loop_depth[index];
        }
    }

  for (int index = 0; index < total_blocks; ++index)
    if (loop_depth[index] > 0
        && static_cast<unsigned long long> (loop_depth[index])
               > snapshot->max_loop_depth)
      snapshot->max_loop_depth =
          static_cast<unsigned long long> (loop_depth[index]);

  /* Dominator-tree height: distance from each block to the entry along
     immediate-dominator links, memoized so shared tails are walked once. */
  std::vector<int> dom_height (total_blocks, -1);
  FOR_ALL_BB_FN (block, fun)
    {
      if (dom_height[block->index] >= 0)
        continue;
      std::vector<basic_block> path;
      basic_block walk = block;
      while (walk != nullptr && dom_height[walk->index] < 0)
        {
          path.push_back (walk);
          walk = get_immediate_dominator (CDI_DOMINATORS, walk);
        }
      int base = walk == nullptr ? 0 : dom_height[walk->index];
      for (auto iterator = path.rbegin (); iterator != path.rend (); ++iterator)
        {
          base += 1;
          dom_height[(*iterator)->index] = base;
        }
    }
  FOR_ALL_BB_FN (block, fun)
    {
      if (block->index < NUM_FIXED_BLOCKS)
        continue;
      if (dom_height[block->index] > 0
          && static_cast<unsigned long long> (dom_height[block->index])
                 > snapshot->dominator_height)
        snapshot->dominator_height =
            static_cast<unsigned long long> (dom_height[block->index]);
      const unsigned long long out_degree = EDGE_COUNT (block->succs);
      if (out_degree > snapshot->max_out_degree)
        snapshot->max_out_degree = out_degree;
    }

  /* McCabe complexity over the same counters the telemetry exports;
     signed arithmetic guards degenerate two-block graphs.  */
  const long long complexity = static_cast<long long> (snapshot->edges)
                               - static_cast<long long> (snapshot->basic_blocks)
                               + 2;
  snapshot->cyclomatic_complexity =
      complexity > 0 ? static_cast<unsigned long long> (complexity) : 0;
}

feature_snapshot
collect_features (function *fun)
{
  feature_snapshot snapshot;

  basic_block block;
  FOR_ALL_BB_FN (block, fun)
    {
      ++snapshot.basic_blocks;
      snapshot.edges += EDGE_COUNT (block->succs);
      for (gphi_iterator phi = gsi_start_phis (block); !gsi_end_p (phi);
           gsi_next (&phi))
        ++snapshot.phi_nodes;
      for (gimple_stmt_iterator statement = gsi_start_bb (block);
           !gsi_end_p (statement); gsi_next (&statement))
        {
          gimple *stmt = gsi_stmt (statement);
          ++snapshot.gimple_statements;
          if (is_gimple_call (stmt))
            ++snapshot.calls;
          if (gimple_code (stmt) == GIMPLE_COND
              || gimple_code (stmt) == GIMPLE_SWITCH)
            ++snapshot.branches;
          if (gimple_vuse (stmt))
            ++snapshot.memory_reads;
          if (gimple_vdef (stmt))
            ++snapshot.memory_writes;
          if (gimple_code (stmt) == GIMPLE_ASSIGN && gimple_num_ops (stmt) > 1)
            {
              tree rhs = gimple_assign_rhs1 (stmt);
              if (rhs && SCALAR_FLOAT_TYPE_P (TREE_TYPE (rhs)))
                ++snapshot.float_ops;
            }
        }
    }
  analyze_structure (fun, &snapshot);
  return snapshot;
}

bool
lookup_feature (const feature_snapshot &snapshot, const std::string &name,
                unsigned long long *value)
{
  if (name == "basic_blocks")
    *value = snapshot.basic_blocks;
  else if (name == "gimple_statements")
    *value = snapshot.gimple_statements;
  else if (name == "phi_nodes")
    *value = snapshot.phi_nodes;
  else if (name == "calls")
    *value = snapshot.calls;
  else if (name == "branches")
    *value = snapshot.branches;
  else if (name == "edges")
    *value = snapshot.edges;
  else if (name == "memory_reads")
    *value = snapshot.memory_reads;
  else if (name == "memory_writes")
    *value = snapshot.memory_writes;
  else if (name == "float_ops")
    *value = snapshot.float_ops;
  else if (name == "back_edges")
    *value = snapshot.back_edges;
  else if (name == "max_loop_depth")
    *value = snapshot.max_loop_depth;
  else if (name == "dominator_height")
    *value = snapshot.dominator_height;
  else if (name == "cyclomatic_complexity")
    *value = snapshot.cyclomatic_complexity;
  else if (name == "max_out_degree")
    *value = snapshot.max_out_degree;
  else
    return false;
  return true;
}

bool
eval_predicate (const predicate &p, const feature_snapshot &snapshot)
{
  unsigned long long observed = 0;
  if (!lookup_feature (snapshot, p.feature, &observed))
    return false;
  switch (p.op)
    {
    case OP_LT: return observed < static_cast<unsigned long long> (p.value);
    case OP_LE: return observed <= static_cast<unsigned long long> (p.value);
    case OP_EQ: return observed == static_cast<unsigned long long> (p.value);
    case OP_NE: return observed != static_cast<unsigned long long> (p.value);
    case OP_GE: return observed >= static_cast<unsigned long long> (p.value);
    case OP_GT: return observed > static_cast<unsigned long long> (p.value);
    }
  return false;
}

bool
eval_rule (const policy_rule &rule, const feature_snapshot &snapshot)
{
  for (const predicate &p : rule.predicates)
    if (!eval_predicate (p, snapshot))
      return false;
  return true;
}

const feature_snapshot *
features_for_current_function ()
{
  if (!current_function_decl)
    return nullptr;
  const auto found = features_by_uid.find (DECL_UID (current_function_decl));
  if (found == features_by_uid.end ())
    return nullptr;
  return &found->second;
}

std::string
trim (const std::string &text)
{
  const auto first = text.find_first_not_of (" \t\r\n");
  if (first == std::string::npos)
    return "";
  const auto last = text.find_last_not_of (" \t\r\n");
  return text.substr (first, last - first + 1);
}

bool
parse_comparison (const std::string &text, comparison_operator *op,
                  size_t *position, size_t *length)
{
  /* Two-character operators first so '<' never consumes '<='.  */
  static const struct
  {
    const char *token;
    comparison_operator op;
    size_t length;
  } operators[] = {
      {"<=", OP_LE, 2}, {">=", OP_GE, 2}, {"==", OP_EQ, 2},
      {"!=", OP_NE, 2}, {"<", OP_LT, 1},  {">", OP_GT, 1},
  };
  bool found = false;
  for (const auto &entry : operators)
    {
      const size_t at = text.find (entry.token);
      if (at != std::string::npos && (!found || at < *position))
        {
          found = true;
          *op = entry.op;
          *position = at;
          *length = entry.length;
        }
    }
  return found;
}

bool
parse_predicate (const std::string &text, predicate *out)
{
  comparison_operator op;
  size_t op_start = 0;
  size_t op_length = 0;
  if (!parse_comparison (text, &op, &op_start, &op_length))
    return false;
  out->feature = trim (text.substr (0, op_start));
  if (out->feature.empty ())
    return false;
  const std::string value_text = trim (text.substr (op_start + op_length));
  if (value_text.empty ())
    return false;
  errno = 0;
  char *end_pointer = nullptr;
  out->value = std::strtoll (value_text.c_str (), &end_pointer, 10);
  if (errno != 0 || end_pointer == value_text.c_str ()
      || *end_pointer != '\0')
    return false;
  out->op = op;
  return true;
}

bool
parse_rule_line (const std::string &raw_line)
{
  const std::string line = trim (raw_line);
  if (line.empty () || line[0] == '#')
    return true;

  const std::string prefix = "disable_pass=";
  if (line.compare (0, prefix.size (), prefix) != 0)
    return false;

  const std::string rest = line.substr (prefix.size ());
  const size_t if_position = rest.find (" if ");
  std::string target =
      trim (if_position == std::string::npos ? rest
                                             : rest.substr (0, if_position));
  if (target.empty ())
    return false;

  /* Optional '#N' suffix restricts the rule to one static instance of the
     pass; instance numbers are visible in pass-gate telemetry events.  */
  policy_rule rule;
  const size_t hash_position = target.find ('#');
  if (hash_position != std::string::npos)
    {
      const std::string number_text = target.substr (hash_position + 1);
      target = trim (target.substr (0, hash_position));
      if (number_text.empty () || !std::all_of (number_text.begin (),
                                                number_text.end (), ::isdigit))
        return false;
      errno = 0;
      char *end_pointer = nullptr;
      const long parsed = std::strtol (number_text.c_str (), &end_pointer, 10);
      if (errno != 0 || end_pointer == number_text.c_str ()
          || *end_pointer != '\0' || parsed < 0)
        return false;
      rule.instance = static_cast<int> (parsed);
    }
  rule.pass_name = target;

  if (rule.pass_name.empty ()
      || rule.pass_name.find_first_of (" \t#") != std::string::npos)
    return false;

  if (if_position != std::string::npos)
    {
      rule.global = false;
      const std::string conditions = rest.substr (if_position + 4);
      size_t start = 0;
      while (true)
        {
          const size_t separator = conditions.find ("&&", start);
          const std::string chunk = trim (
              separator == std::string::npos
                  ? conditions.substr (start)
                  : conditions.substr (start, separator - start));
          predicate parsed;
          if (!parse_predicate (chunk, &parsed))
            return false;
          rule.predicates.push_back (parsed);
          if (separator == std::string::npos)
            break;
          start = separator + 2;
        }
    }

  for (const policy_rule &existing : rules)
    if (existing.pass_name == rule.pass_name
        && existing.instance == rule.instance
        && existing.global == rule.global
        && existing.predicates.size () == rule.predicates.size ())
      {
        bool identical = true;
        for (size_t index = 0; index < rule.predicates.size (); ++index)
          if (existing.predicates[index].feature != rule.predicates[index].feature
              || existing.predicates[index].op != rule.predicates[index].op
              || existing.predicates[index].value != rule.predicates[index].value)
            {
              identical = false;
              break;
            }
        if (identical)
          return true;
      }

  rules.push_back (rule);
  return true;
}

const pass_data ai_observe_pass_data = {
  GIMPLE_PASS,
  "ai_native_observe",
  OPTGROUP_NONE,
  TV_NONE,
  PROP_cfg | PROP_ssa,
  0,
  0,
  0,
  0,
};

class ai_observe_pass final : public gimple_opt_pass
{
public:
  explicit ai_observe_pass (gcc::context *context)
    : gimple_opt_pass (ai_observe_pass_data, context)
  {}

  unsigned int execute (function *fun) final
  {
    const feature_snapshot snapshot = collect_features (fun);
    features_by_uid[DECL_UID (fun->decl)] = snapshot;

    tree declaration = fun->decl;
    location_t location = DECL_SOURCE_LOCATION (declaration);
    std::ostringstream event;
    event << "{\"schema\":\"gcc-ai.telemetry.v3\","
          << "\"event\":\"function-ir\","
          << "\"function\":\"" << json_escape (decl_name (declaration)) << "\","
          << "\"source\":\"" << json_escape (LOCATION_FILE (location)) << "\","
          << "\"line\":" << LOCATION_LINE (location) << ','
          << "\"basic_blocks\":" << snapshot.basic_blocks << ','
          << "\"gimple_statements\":" << snapshot.gimple_statements << ','
          << "\"phi_nodes\":" << snapshot.phi_nodes << ','
          << "\"calls\":" << snapshot.calls << ','
          << "\"branches\":" << snapshot.branches << ','
          << "\"edges\":" << snapshot.edges << ','
          << "\"memory_reads\":" << snapshot.memory_reads << ','
          << "\"memory_writes\":" << snapshot.memory_writes << ','
          << "\"float_ops\":" << snapshot.float_ops << ','
          << "\"back_edges\":" << snapshot.back_edges << ','
          << "\"max_loop_depth\":" << snapshot.max_loop_depth << ','
          << "\"dominator_height\":" << snapshot.dominator_height << ','
          << "\"cyclomatic_complexity\":" << snapshot.cyclomatic_complexity
          << ','
          << "\"max_out_degree\":" << snapshot.max_out_degree << "}";
    write_event (event.str ());

    if (cfg_export)
      write_event (cfg_event (fun));
    return 0;
  }

  ai_observe_pass *clone () final { return new ai_observe_pass (g); }
};

bool
rule_targets_current_instance (const policy_rule &rule,
                               const std::string &pass_name)
{
  if (rule.pass_name != pass_name)
    return false;
  /* A rule without an explicit '#N' suffix targets every instance of the
     named pass; with a suffix it matches exactly that static instance.  */
  return rule.instance < 0
         || rule.instance == current_pass->static_pass_number;
}

void
override_pass_gate (void *gcc_data, void *)
{
  int *gate = static_cast<int *> (gcc_data);
  if (!gate || !*gate || !current_pass || !current_pass->name)
    return;
  const std::string pass_name (current_pass->name);

  std::string scope;
  for (const policy_rule &rule : rules)
    {
      if (rule.global && rule_targets_current_instance (rule, pass_name))
        {
          scope = "global";
          break;
        }
    }
  if (scope.empty ())
    {
      const feature_snapshot *snapshot = features_for_current_function ();
      if (snapshot)
        for (const policy_rule &rule : rules)
          {
            if (rule.global
                || !rule_targets_current_instance (rule, pass_name))
              continue;
            const bool matched = eval_rule (rule, *snapshot);
            if (audit_all_evaluations)
              {
                std::ostringstream eval_event;
                eval_event << "{\"schema\":\"gcc-ai.telemetry.v3\","
                           << "\"event\":\"gate-eval\","
                           << "\"pass\":\"" << json_escape (pass_name.c_str ())
                           << "\","
                           << "\"instance\":"
                           << current_pass->static_pass_number << ','
                           << "\"function\":"
                           << "\"" << json_escape (decl_name (current_function_decl))
                           << "\","
                           << "\"matched\":" << (matched ? "true" : "false") << "}";
                write_event (eval_event.str ());
              }
            if (matched)
              {
                scope = "rule";
                break;
              }
          }
    }

  if (scope.empty ())
    return;

  *gate = 0;
  std::ostringstream event;
  event << "{\"schema\":\"gcc-ai.telemetry.v3\","
        << "\"event\":\"pass-gate\","
        << "\"pass\":\"" << json_escape (current_pass->name) << "\","
        << "\"instance\":" << current_pass->static_pass_number << ','
        << "\"function\":\"" << json_escape (decl_name (current_function_decl))
        << "\","
        << "\"scope\":\"" << scope << "\","
        << "\"decision\":\"disabled-by-policy\"}";
  write_event (event.str ());
}

void
finish_plugin (void *, void *)
{
  if (output.is_open ())
    output.close ();
}

bool
load_policy (const char *path)
{
  std::ifstream policy (path);
  if (!policy)
    return false;
  std::string line;
  while (std::getline (policy, line))
    if (!parse_rule_line (line))
      {
        error ("AI-native GCC plugin found malformed policy line %qs in %qs",
               line.c_str (), path);
        return false;
      }
  return true;
}

} // namespace

int
plugin_init (plugin_name_args *arguments, plugin_gcc_version *version)
{
  plugin_name = arguments->base_name;
  if (!plugin_default_version_check (version, &gcc_version))
    return 1;

  for (int index = 0; index < arguments->argc; ++index)
    {
      const plugin_argument &argument = arguments->argv[index];
      const std::string key = argument.key ? argument.key : "";
      const char *value = argument.value;
      if (key == "output" && value)
        output_path = value;
      else if (key == "policy" && value)
        {
          if (!load_policy (value))
            {
              error ("AI-native GCC plugin could not read policy file %qs", value);
              return 1;
            }
        }
      else if (key == "audit" && value)
        {
          const std::string mode (value);
          if (mode == "all")
            audit_all_evaluations = true;
          else if (mode == "gates")
            audit_all_evaluations = false;
          else
            {
              error ("unknown AI-native audit mode %qs; expected gates or all",
                     value);
              return 1;
            }
        }
      else if (key == "cfg" && value)
        {
          const std::string mode (value);
          if (mode == "export")
            cfg_export = true;
          else if (mode == "none")
            cfg_export = false;
          else
            {
              error ("unknown AI-native cfg mode %qs; expected export or none",
                     value);
              return 1;
            }
        }
      else
        {
          error ("unknown AI-native GCC plugin argument %qs", argument.key);
          return 1;
        }
    }

  if (!output_path.empty ())
    {
      output.open (output_path, std::ios::out | std::ios::app);
      if (!output)
        {
          error ("AI-native GCC plugin could not open telemetry output %qs",
                 output_path.c_str ());
          return 1;
        }
    }

  register_pass_info pass_info;
  pass_info.pass = new ai_observe_pass (g);
  pass_info.reference_pass_name = "ssa";
  pass_info.ref_pass_instance_number = 1;
  pass_info.pos_op = PASS_POS_INSERT_AFTER;
  register_callback (plugin_name, PLUGIN_PASS_MANAGER_SETUP, nullptr,
                     &pass_info);
  register_callback (plugin_name, PLUGIN_OVERRIDE_GATE, override_pass_gate,
                     nullptr);
  register_callback (plugin_name, PLUGIN_FINISH, finish_plugin, nullptr);

  static struct plugin_info metadata = {
    "0.3.0",
    "Compiler-native IR telemetry and conservative learned pass policy hook",
  };
  register_callback (plugin_name, PLUGIN_INFO, nullptr, &metadata);
  return 0;
}
