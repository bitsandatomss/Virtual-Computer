"""Dataset census: per-program effects + worst unstable rows."""
import json
from collections import defaultdict

rows = [json.loads(line) for line in open(
    r"C:\Users\aacer\Documents\Virtual Compiler\experiments\study\dataset.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["program"]].append(r)
for prog, rs in sorted(by.items()):
    ts = [(r["outcome"]["runtime_ms"], r["flags"], r.get("xform"))
          for r in rs
          if r["outcome"]["build_ok"] and r["outcome"]["runtime_ms"]]
    ts.sort()
    o0 = next(t for t in ts if list(t[1]) == ["-O0"])
    o3 = next(t for t in ts if list(t[1]) == ["-O3"])
    print(f"{prog}: n={len(ts)} O0={o0[0]:.0f} O3={o3[0]:.0f} "
          f"O0/O3={o0[0] / o3[0]:.2f} best={ts[0][0]:.0f}{list(ts[0][1])} "
          f"worst={ts[-1][0]:.0f}")
print()
print("worst-spread rows:")
spr = []
for r in rows:
    s = r["outcome"].get("samples") or []
    m = r["outcome"]["runtime_ms"]
    if len(s) >= 3 and m:
        spr.append(((max(s) - min(s)) / m, r["program"], r["flags"],
                    [round(x) for x in s]))
spr.sort(reverse=True)
for x in spr[:8]:
    print(f"  spread={x[0]:.2f} {x[1]} {x[2]} samples={x[3]}")
