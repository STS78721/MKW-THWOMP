// Optional scalar-only observation of unsupported NVN2 modes and recording boundaries.
#pragma once
#include <stdint.h>

enum {
    NVN_MODE_INIT, NVN_MODE_FINALIZE, NVN_MODE_BEGIN, NVN_MODE_END,
    NVN_MODE_RECORDING, NVN_MODE_GEOMETRY, NVN_MODE_PREPARE, NVN_MODE_HANDLE,
    NVN_MODE_EVENT_COUNT
};
#define NVN_MODE_TRACE_LIMIT 65536u

// Tokens are opaque values already available to the caller, never dereferenced.
// Sequence/thread identify observations, not guest or GPU happens-before edges.
// Optional restart settings START_MS/DURATION_MS select elapsed host time from
// the first enabled hook. Start is inclusive; end exclusive; duration 0 is open.
void nvn_mode_trace(unsigned kind, uint64_t command_buffer, uint64_t handle, uint32_t value, uint64_t submit);
// Terminal snapshot: seals once and emits the bounded stream plus exact counters.
// Called by prof_write on the existing process-exit paths, or by normal atexit.
void nvn_mode_trace_finish(void);
