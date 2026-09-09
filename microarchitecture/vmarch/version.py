"""Version contracts. Bump when semantics change so old checkpoints/benchmarks
are rejected instead of silently miscompared (DEA methodology)."""
VMARCH_VERSION = "0.1.0"
FEATURE_VERSION = 1        # observation feature schema
MODEL_FORMAT_VERSION = 1   # surrogate checkpoint schema
BENCH_PROTOCOL_VERSION = 1 # benchmark artifact schema
