#include "wgpu_timing.h"

#include <webgpu/wgpu.h>

#include <algorithm>

#include "xt/wgpu/wgpu_device.h"

/// T-734 (Docs/design/10-backends.md §10.19): GpuTimer (wgpu_timing.h) and the three WgpuDevice
/// hooks through which `submit` creates its frame encoder, begins its render passes and finishes
/// the encoder. Without a timer (the default) the hooks are the plain wgpu calls.

namespace xt::wgpu {

namespace detail {

GpuTimer::GpuTimer() = default;

GpuTimer::~GpuTimer() { disable(); }

bool GpuTimer::enable(WGPUDevice device, WGPUQueue queue) {
    if (enabled()) {
        return true;
    }
    if (device == nullptr || queue == nullptr ||
        wgpuDeviceHasFeature(device, WGPUFeatureName_TimestampQuery) == 0) {
        return false;
    }
    m_device = device;
    m_queue = queue;
    wgpuDeviceAddRef(m_device);
    wgpuQueueAddRef(m_queue);
    m_encoderLevel =
        wgpuDeviceHasFeature(
            device, static_cast<WGPUFeatureName>(WGPUNativeFeature_TimestampQueryInsideEncoders)) != 0;
    // wgpu-native: raw ticks unless the instance has AutomaticTimestampNormalization (then 1.0);
    // ticks x period = nanoseconds either way.
    m_periodNs = static_cast<double>(wgpuQueueGetTimestampPeriod(queue));
    for (std::unique_ptr<Slot>& s : m_slots) {
        s = std::make_unique<Slot>();
        s->owner = this;
    }
    return true;
}

void GpuTimer::releaseBuffers() {
    if (m_querySet != nullptr) {
        wgpuQuerySetRelease(
            m_querySet);  // wgpu-native: QuerySetDestroy already drops it (a Release after it panics)
        m_querySet = nullptr;
    }
    if (m_resolve != nullptr) {
        wgpuBufferDestroy(m_resolve);
        wgpuBufferRelease(m_resolve);
        m_resolve = nullptr;
    }
    m_capacity = 0;
}

void GpuTimer::disable() {
    if (!enabled()) {
        return;
    }
    // Shutdown only: let outstanding maps call back before their slots go.
    for (int i = 0; i < 100; ++i) {
        bool pending = false;
        for (const std::unique_ptr<Slot>& s : m_slots) {
            pending = pending || (s && s->state.load() == kMapping);
        }
        if (!pending) break;
        wgpuDevicePoll(m_device, 1, nullptr);
    }
    for (std::unique_ptr<Slot>& s : m_slots) {
        if (!s) continue;
        if (s->state.load() == kMapping) {
            s->owner = nullptr;
            (void)s.release();  // its callback may still come: the slot must stay valid (leaked)
            continue;
        }
        if (s->buffer != nullptr) {
            wgpuBufferDestroy(s->buffer);
            wgpuBufferRelease(s->buffer);
        }
        s.reset();
    }
    releaseBuffers();
    m_active = nullptr;
    m_encoded = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_results.clear();
    }
    wgpuQueueRelease(m_queue);
    wgpuDeviceRelease(m_device);
    m_queue = nullptr;
    m_device = nullptr;
    m_encoderLevel = false;
}

void GpuTimer::beginFrame(uint64_t frame, size_t gpuPasses) {
    m_active = nullptr;
    m_encoded = false;
    m_overflow = false;
    m_passes = 0;
    if (!enabled()) {
        return;
    }
    Slot* slot = nullptr;
    for (std::unique_ptr<Slot>& s : m_slots) {
        if (s && s->state.load() == kFree) {
            slot = s.get();
            break;
        }
    }
    if (slot == nullptr) {
        ++m_dropped;  // every slot is still being read back: this frame is not timed
        return;
    }
    // A gpu::Pass encodes at most a clear (or load) pass and its own pass (wgpu_device.cpp
    // encodePass); one more for a frame that only clears the surface. 3 per pass leaves headroom.
    const uint64_t want = (m_encoderLevel ? 2u : 0u) + 2u * (3u * static_cast<uint64_t>(gpuPasses) + 1u);
    const uint32_t needed = static_cast<uint32_t>(std::min<uint64_t>(want, kMaxQueries));
    if (needed > m_capacity) {
        releaseBuffers();
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.label = WGPUStringView{"xt_wgpu timestamps", WGPU_STRLEN};
        qd.type = WGPUQueryType_Timestamp;
        qd.count = needed;
        m_querySet = wgpuDeviceCreateQuerySet(m_device, &qd);
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.label = WGPUStringView{"xt_wgpu timestamp resolve", WGPU_STRLEN};
        bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
        bd.size = static_cast<uint64_t>(needed) * 8u;
        m_resolve = wgpuDeviceCreateBuffer(m_device, &bd);
        if (m_querySet == nullptr || m_resolve == nullptr) {
            releaseBuffers();
            ++m_dropped;
            return;
        }
        m_capacity = needed;
    }
    slot->frame = frame;
    slot->encoderLevel = m_encoderLevel;
    m_active = slot;
}

void GpuTimer::encoderBegin(WGPUCommandEncoder enc) {
    if (m_active == nullptr || m_encoded) {
        return;
    }
    m_encoded = true;
    if (m_encoderLevel) {
        wgpuCommandEncoderWriteTimestamp(enc, m_querySet, 0);
    }
}

const WGPUPassTimestampWrites* GpuTimer::passWrites() {
    if (m_active == nullptr || !m_encoded) {
        return nullptr;
    }
    const uint32_t first = (m_encoderLevel ? 2u : 0u) + 2u * m_passes;
    if (first + 2u > m_capacity) {
        m_overflow = true;
        return nullptr;
    }
    ++m_passes;
    m_writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    m_writes.querySet = m_querySet;
    m_writes.beginningOfPassWriteIndex = first;
    m_writes.endOfPassWriteIndex = first + 1u;
    return &m_writes;
}

void GpuTimer::encoderEnd(WGPUCommandEncoder enc) {
    if (m_active == nullptr || !m_encoded) {
        return;
    }
    if (m_passes == 0 || m_overflow) {
        if (m_overflow) ++m_dropped;
        m_active = nullptr;  // no result (see the file comment); the encoder start stamp is unused
        return;
    }
    if (m_encoderLevel) {
        wgpuCommandEncoderWriteTimestamp(enc, m_querySet, 1);
    }
    Slot& s = *m_active;
    const uint32_t queries = (m_encoderLevel ? 2u : 0u) + 2u * m_passes;
    const uint64_t bytes = static_cast<uint64_t>(queries) * 8u;
    if (s.buffer == nullptr || s.size < bytes) {
        if (s.buffer != nullptr) {
            wgpuBufferDestroy(s.buffer);
            wgpuBufferRelease(s.buffer);
        }
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.label = WGPUStringView{"xt_wgpu timestamp readback", WGPU_STRLEN};
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        bd.size = static_cast<uint64_t>(m_capacity) * 8u;
        s.buffer = wgpuDeviceCreateBuffer(m_device, &bd);
        s.size = s.buffer != nullptr ? bd.size : 0;
        if (s.buffer == nullptr) {
            ++m_dropped;
            m_active = nullptr;
            return;
        }
    }
    wgpuCommandEncoderResolveQuerySet(enc, m_querySet, 0, queries, m_resolve, 0);
    wgpuCommandEncoderCopyBufferToBuffer(enc, m_resolve, 0, s.buffer, 0, bytes);
    s.queries = queries;
    s.passes = m_passes;
    s.state.store(kEncoded);
}

void GpuTimer::endFrame() {
    Slot* s = m_active;
    m_active = nullptr;
    m_encoded = false;
    if (s == nullptr || s->state.load() != kEncoded) {
        return;
    }
    s->state.store(kMapping);
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = &GpuTimer::onMapped;
    cb.userdata1 = s;
    cb.userdata2 = nullptr;
    wgpuBufferMapAsync(s->buffer, WGPUMapMode_Read, 0, static_cast<size_t>(s->queries) * 8u, cb);
}

void GpuTimer::onMapped(WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
    Slot* s = static_cast<Slot*>(userdata1);
    GpuTimer* t = s->owner;  // null: the timer was disabled while this map was outstanding
    if (status == WGPUMapAsyncStatus_Success) {
        const size_t bytes = static_cast<size_t>(s->queries) * 8u;
        const auto* ts = static_cast<const uint64_t*>(wgpuBufferGetConstMappedRange(s->buffer, 0, bytes));
        if (ts != nullptr && t != nullptr) {
            auto ms = [t](uint64_t a, uint64_t b) {
                return b > a ? static_cast<double>(b - a) * t->m_periodNs / 1.0e6 : 0.0;
            };
            WgpuGpuTiming r;
            r.frame = s->frame;
            r.passes = s->passes;
            r.encoderLevel = s->encoderLevel;
            const uint32_t base = s->encoderLevel ? 2u : 0u;
            for (uint32_t k = 0; k < s->passes; ++k) {
                r.passSumMs += ms(ts[base + 2u * k], ts[base + 2u * k + 1u]);
            }
            r.spanMs = s->encoderLevel ? ms(ts[0], ts[1]) : ms(ts[0], ts[base + 2u * s->passes - 1u]);
            t->push(r);
        }
        wgpuBufferUnmap(s->buffer);
    }
    s->state.store(kFree);
}

void GpuTimer::push(const WgpuGpuTiming& t) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_results.size() >= kMaxResults) {
        m_results.pop_front();  // the oldest unread result goes
        ++m_dropped;
    }
    m_results.push_back(t);
}

size_t GpuTimer::take(std::vector<WgpuGpuTiming>& out) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t n = m_results.size();
    out.insert(out.end(), m_results.begin(), m_results.end());
    m_results.clear();
    return n;
}

}  // namespace detail

WGPUCommandEncoder WgpuDevice::createFrameEncoder() {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(m_device, nullptr);
    if (m_timer != nullptr && enc != nullptr) m_timer->encoderBegin(enc);
    return enc;
}

WGPURenderPassEncoder WgpuDevice::beginRenderPass(WGPUCommandEncoder enc, WGPURenderPassDescriptor& rp) {
    if (m_timer != nullptr) rp.timestampWrites = m_timer->passWrites();
    return wgpuCommandEncoderBeginRenderPass(enc, &rp);
}

WGPUCommandBuffer WgpuDevice::finishFrameEncoder(WGPUCommandEncoder enc) {
    if (m_timer != nullptr) m_timer->encoderEnd(enc);
    return wgpuCommandEncoderFinish(enc, nullptr);
}

}  // namespace xt::wgpu
