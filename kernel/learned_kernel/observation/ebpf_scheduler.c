#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Per-CPU aggregate counters (exact; readable from user space any time).
BPF_ARRAY(cpu_metrics, u64, 256);
// Per-CPU event sampler state.
BPF_ARRAY(sample_state, u64, 256);
// Sampled sched_switch events streamed to user space. Submitting on EVERY
// switch floods perf buffers on busy hosts (tens of thousands of events/s),
// so we stream a 1-in-N sample; the aggregate counters stay exact.
BPF_PERF_OUTPUT(events);

#define SAMPLE_PERIOD 16

struct sched_event_data {
    u32 prev_pid;
    u32 next_pid;
    u32 cpu;
    u64 ts;
};

// Hook on sched:sched_switch tracepoint to observe scheduler decisions.
TRACEPOINT_PROBE(sched, sched_switch) {
    u32 cpu = bpf_get_smp_processor_id();
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;
    u64 ts = bpf_ktime_get_ns();

    // Exact aggregate increment (never sampled).
    u64 zero = 0, *count;
    count = cpu_metrics.lookup_or_init(&cpu, &zero);
    if (count) {
        (*count)++;
    }

    // Sampled streaming to the Python aggregator.
    u64 *seq = sample_state.lookup_or_init(&cpu, &zero);
    if (!seq) {
        return 0;
    }
    (*seq)++;
    if (*seq % SAMPLE_PERIOD != 1) {
        return 0;
    }

    struct sched_event_data event = {};
    event.prev_pid = prev_pid;
    event.next_pid = next_pid;
    event.cpu = cpu;
    event.ts = ts;
    events.perf_submit(args, &event, sizeof(event));

    return 0;
}
