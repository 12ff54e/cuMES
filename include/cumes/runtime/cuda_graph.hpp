// cuda_graph.hpp — CUDA Graph capture/instantiate/launch RAII (blueprint
// §8.11).
//
// A minimal owning wrapper over the CUDA Graph runtime API: capture a stream's
// enqueued work into a cudaGraph_t, instantiate it into a cudaGraphExec_t, and
// launch the executable. It is the Phase-9 measurement primitive for the
// "graphs reduce submission overhead" question — not (yet) a solver backend.
//
// Capture is stream-scoped and thread-local; the caller enqueues the DAG on the
// capturing stream inside the callback. On a throwing callback the partial
// capture is torn down so the stream is not left in capture mode. Launching a
// graph replays the captured work in one submission.
#pragma once

#include "cumes/runtime/cuda_status.hpp"

#include <cuda_runtime.h>

namespace cumes {

class CudaGraph {
   public:
    CudaGraph() = default;

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;

    CudaGraph(CudaGraph&& other) noexcept
        : graph_(other.graph_), exec_(other.exec_) {
        other.graph_ = nullptr;
        other.exec_ = nullptr;
    }

    CudaGraph& operator=(CudaGraph&& other) noexcept {
        if (this != &other) {
            reset();
            graph_ = other.graph_;
            exec_ = other.exec_;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
        }
        return *this;
    }

    ~CudaGraph() { reset(); }

    // Begin capture on `stream`, run `enqueue` (which must only enqueue work on
    // that stream), end capture, and instantiate. Throws CumesError on any CUDA
    // failure; a throwing `enqueue` tears down the partial capture and
    // rethrows.
    template <class Fn>
    static CudaGraph capture(cudaStream_t stream, Fn&& enqueue) {
        CudaGraph g;
        check_cuda(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "CudaGraph::capture begin");
        try {
            enqueue();
        } catch (...) {
            cudaGraph_t partial = nullptr;
            cudaStreamEndCapture(stream, &partial);  // best-effort teardown
            if (partial != nullptr) cudaGraphDestroy(partial);
            throw;
        }
        check_cuda(cudaStreamEndCapture(stream, &g.graph_),
                   "CudaGraph::capture end");
        check_cuda(
            cudaGraphInstantiate(&g.exec_, g.graph_, nullptr, nullptr, 0),
            "CudaGraph::capture instantiate");
        return g;
    }

    void launch(cudaStream_t stream) const {
        if (exec_ == nullptr) {
            throw CumesError("CudaGraph::launch on an empty graph");
        }
        check_cuda(cudaGraphLaunch(exec_, stream), "CudaGraph::launch");
    }

    bool empty() const { return exec_ == nullptr; }

    void reset() {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
    }

   private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

}  // namespace cumes
