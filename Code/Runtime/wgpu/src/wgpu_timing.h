#pragma once

#include <webgpu/webgpu.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "xt/wgpu/wgpu_inflight.h"
#include "xt/wgpu/wgpu_renderer.h"

/// T-734 (Docs/design/10-backends.md §10.19): the GPU timestamps of a renderer's frames, a
/// diagnostic that is off by default. While off it owns nothing and writes nothing.
///
/// One frame = the command encoder of the renderer's `WgpuDevice::submit` (the main frame submit;
/// world-layer submits made during the build and mip generation are separate encoders and are not
/// timed). Query layout: with encoder timestamps (`WGPUNativeFeature_TimestampQueryInsideEncoders`)
/// index 0 / 1 are the encoder's start / end and pass k writes 2 + 2k / 3 + 2k; without them pass k
/// writes 2k / 2k + 1. The query set is resolved into a QueryResolve buffer at the end of the
/// encoder and copied into a MapRead slot of a ring (WgpuFramesInFlight::kMaxFrames + 2 slots); the
/// slot is mapped after the queue submit and its map callback turns the ticks (x
/// `wgpuQueueGetTimestampPeriod`) into a `WgpuGpuTiming`. Nothing here waits for the GPU, except
/// `disable` (shutdown), which lets outstanding maps finish.
///
/// A frame that encodes no render pass gives no result. A frame that finds every slot still mapping,
/// or encodes more passes than the query set holds, is not timed and counts in `dropped`, as does an
/// unread result pushed out of the full result queue (oldest first).

namespace xt::wgpu::detail {

class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    /// Turns timing on for `device` / `queue`. False (and stays off) without TimestampQuery.
    bool enable(WGPUDevice device, WGPUQueue queue);
    /// Releases the query set and buffers (waits for outstanding maps first). Safe to call twice.
    void disable();
    bool enabled() const { return m_device != nullptr; }
    bool encoderLevel() const { return m_encoderLevel; }
    /// For tests: a query set exists.
    bool hasQuerySet() const { return m_querySet != nullptr; }

    /// Arms the next frame encoder: frame number `frame`, at most `gpuPasses` gpu::Pass entries.
    void beginFrame(uint64_t frame, size_t gpuPasses);
    /// WgpuDevice hooks (no-ops unless armed): the encoder was created; the timestamp writes for the
    /// render pass about to begin (null = not timed); the encoder is about to be finished.
    void encoderBegin(WGPUCommandEncoder enc);
    const WGPUPassTimestampWrites* passWrites();
    void encoderEnd(WGPUCommandEncoder enc);
    /// After the queue submit: maps the frame's slot. Disarms.
    void endFrame();

    size_t take(std::vector<WgpuGpuTiming>& out);
    uint64_t dropped() const { return m_dropped.load(); }

private:
    enum SlotState : int { kFree = 0, kEncoded = 1, kMapping = 2 };
    struct Slot {
        GpuTimer* owner = nullptr;
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;
        std::atomic<int> state{kFree};
        uint64_t frame = 0;
        uint32_t queries = 0;
        uint32_t passes = 0;
        bool encoderLevel = false;
    };
    static constexpr size_t kSlots = WgpuFramesInFlight::kMaxFrames + 2;
    static constexpr uint32_t kMaxQueries = 4096;  ///< wgpu's QUERY_SET_MAX_QUERIES
    static constexpr size_t kMaxResults = 1024;

    static void onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2);
    void push(const WgpuGpuTiming& t);
    void releaseBuffers();

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    bool m_encoderLevel = false;
    double m_periodNs = 1.0;
    WGPUQuerySet m_querySet = nullptr;
    WGPUBuffer m_resolve = nullptr;
    uint32_t m_capacity = 0;
    std::array<std::unique_ptr<Slot>, kSlots> m_slots;

    // The armed frame.
    Slot* m_active = nullptr;
    bool m_encoded = false;
    bool m_overflow = false;
    uint32_t m_passes = 0;
    WGPUPassTimestampWrites m_writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;

    std::mutex m_mutex;  ///< m_results (map callbacks may run on a thread that polls the device)
    std::deque<WgpuGpuTiming> m_results;
    std::atomic<uint64_t> m_dropped{0};
};

}  // namespace xt::wgpu::detail
