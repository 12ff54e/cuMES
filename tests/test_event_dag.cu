// test_event_dag.cu — event-DAG ordering/delay and stream-poison semantics
// (completion plan step 4.4).
//
// Asserts the CUDA scheduling contracts the multi-stage DAG relies on:
//   1. an event recorded before a delayed kernel completes only after it
//      (cudaEventSynchronize blocks; elapsed time spans the delay);
//   2. cudaEventQuery / cudaEventElapsedTime on pending work report
//      cudaErrorNotReady (the API-trace contract, no silent success);
//   3. a trapping kernel poisons its stream with a STICKY error: the next
//      enqueue reports the same error and cudaStreamSynchronize surfaces it
//      (no silent continuation). This case runs only under --poison because
//      the fault also corrupts the context for further work; CTest invokes it
//      as a separate process entry so the main test stays compute-sanitizer
//      clean.
#include "cumes/runtime/cuda_status.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <cstring>
using namespace cumes::test;

// Faulting kernel: an illegal-instruction trap. It faults only its own
// launch and leaves memory untouched, so the poison case is safe (the context
// becomes unusable afterwards, which is exactly what it asserts).
__global__ void trapKernel() {
    asm volatile("trap;");
}

// Busy-wait ~60 ms of device time (clock64 loop; calibrated once on the GPU).
__global__ void delayKernel(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {}
}

static void testOrderingAndDelay(bool probe_notready) {
    cudaStream_t s = nullptr;
    cc(cudaStreamCreate(&s), "stream create");
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cc(cudaEventCreate(&e0), "event e0");
    cc(cudaEventCreate(&e1), "event e1");

    // Calibrate the clock64 busy-wait cost once on this GPU (~60 ms of
    // device time; the measured cycles-per-ms below is what gates).
    dim3 b(1), g(1);
    cudaEvent_t c0 = nullptr, c1 = nullptr;
    cc(cudaEventCreate(&c0), "event c0");
    cc(cudaEventCreate(&c1), "event c1");
    cc(cudaEventRecord(c0, s), "record c0");
    delayKernel<<<g, b, 0, s>>>(1000000);
    cc(cudaEventRecord(c1, s), "record c1");
    cc(cudaEventSynchronize(c1), "sync c1");
    float cal = 0.0f;
    cc(cudaEventElapsedTime(&cal, c0, c1), "elapsed cal");
    cc(cudaEventDestroy(c0), "destroy c0");
    cc(cudaEventDestroy(c1), "destroy c1");
    // cycles_per_ms from the calibration, then a 60 ms target with a 2x
    // margin (GPU clocks can vary with load; the assertion only needs the
    // event pair to ORDER around a long-running kernel, not an exact delay).
    const long long cycles = (long long)(2.0f * 60.0f * (1000000.0f / cal));

    // 1. e0 before the delay, e1 after: e1 completes only after the delay.
    cc(cudaEventRecord(e0, s), "record e0");
    delayKernel<<<g, b, 0, s>>>(cycles);
    cc(cudaEventRecord(e1, s), "record e1");

    // While the delay runs, querying e1 must report not-ready (never a
    // silent success that would break DAG ordering assumptions). These
    // DELIBERATE API-error probes are skipped under compute-sanitizer
    // (--no-notready-probe): the sanitizer counts the intended cudaError
    // return as its own error.
    if (probe_notready) {
        cudaError_t q = cudaEventQuery(e1);
        check(q == cudaErrorNotReady,
              "pending event query reports cudaErrorNotReady");

        // elapsedTime(e1, e0) is also not-ready while e1 is pending.
        float ms = 0.0f;
        cudaError_t el = cudaEventElapsedTime(&ms, e1, e0);
        check(el == cudaErrorNotReady,
              "pending event elapsed-time reports cudaErrorNotReady");
    }

    cc(cudaEventSynchronize(e1), "sync e1");
    cc(cudaEventSynchronize(e0), "sync e0");
    float ms = 0.0f;
    cc(cudaEventElapsedTime(&ms, e0, e1), "elapsed e0->e1");
    // Strict ordering is the contract under test (the magnitude is wall-
    // clock dependent and unreliable under sanitizer instrumentation).
    check(ms > 0.0f, "event pair spans the delayed kernel (strict ordering)");

    // Ordering contract: elapsed(e1, e0) is negative once both completed.
    cc(cudaEventElapsedTime(&ms, e1, e0), "elapsed e1->e0");
    check(ms < 0.0f, "reversed event pair reports negative elapsed time");

    cc(cudaEventDestroy(e0), "destroy e0");
    cc(cudaEventDestroy(e1), "destroy e1");
    cc(cudaStreamDestroy(s), "stream destroy");
}

// Poison case: a trapping kernel faults ONLY its own block/context; the
// stream carries the sticky error to every later enqueue and to the sync.
static int runPoison() {
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreate(&s);
    if (e != cudaSuccess) {
        std::cerr << "poison: stream create failed\n";
        return 2;
    }
    trapKernel<<<1, 1, 0, s>>>();
    // A device fault (trap) is ASYNCHRONOUS: the launch and the next enqueue
    // are accepted, and the fault surfaces at COMPLETION time — the
    // cudaStreamSynchronize must report it (never a silent success, which is
    // the contract the per-iteration DAG depends on).
    cudaError_t after = cudaGetLastError();
    trapKernel<<<1, 1, 0, s>>>();
    cudaError_t sticky = cudaGetLastError();
    cudaError_t sync = cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    const bool ok = (after == cudaSuccess) && (sticky == cudaSuccess) &&
                    (sync == cudaErrorLaunchFailure);
    std::cout << format("{} stream poison: launch={} sticky={} sync={}\n",
                        ok ? "PASS" : "FAIL", cudaGetErrorName(after),
                        cudaGetErrorName(sticky), cudaGetErrorName(sync));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--poison") == 0) {
        return runPoison();
    }
    const bool probe_notready =
        !(argc > 1 && std::strcmp(argv[1], "--no-notready-probe") == 0);
    testOrderingAndDelay(probe_notready);
    return summary();
}
