#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "wgpu_internal.h"

/// The frame's geometry buffers (T-438, Docs/design/10-backends.md §10.10.2 first row), each written
/// against its own CPU shadow so only changed element runs go through wgpuQueueWriteBuffer --
/// gl_geometry.cpp's T-159 strategy with WebGPU writes. T-726: one set, compared with the previous
/// frame. wgpuQueueWriteBuffer is a queue operation: it takes effect after every command buffer
/// submitted before it and before every one submitted after it, so a write for the next frame never
/// changes what a frame still in flight reads; a buffer replaced by a larger one is destroyed by wgpu
/// only once the submissions using it are done -- the single uniform buffer has always worked this way.

namespace xt::wgpu::detail {

namespace {
/// T-726: two changed runs closer than this many bytes are written as one; XT_WGPU_UPLOAD_MERGE_GAP
/// overrides it (bytes, read once). Measured on War Table: 256 B / 64 B write 7-15 % fewer bytes than
/// 1 KB but cost 0.15-0.5 ms more CPU per frame in the extra writeBuffer calls.
constexpr size_t kDefaultMergeGapBytes = 1024;
/// Buffers grow in this many bytes.
constexpr uint64_t kCapacityStep = 64 * 1024;

size_t mergeGapBytes() {
    static const size_t gap = [] {
#if defined(_MSC_VER)
        size_t len = 0;
        char buf[32] = {0};
        if (getenv_s(&len, buf, sizeof(buf), "XT_WGPU_UPLOAD_MERGE_GAP") != 0 || len <= 1) {
            return kDefaultMergeGapBytes;
        }
        const char* text = buf;
#else
        const char* text = std::getenv("XT_WGPU_UPLOAD_MERGE_GAP");
        if (text == nullptr || *text == '\0') {
            return kDefaultMergeGapBytes;
        }
#endif
        char* end = nullptr;
        const unsigned long v = std::strtoul(text, &end, 10);
        return (end != text && *end == '\0') ? static_cast<size_t>(v) : kDefaultMergeGapBytes;
    }();
    return gap;
}
}  // namespace

void ShadowBuffer::release() {
    if (buffer != nullptr) {
        wgpuBufferDestroy(buffer);
        wgpuBufferRelease(buffer);
        buffer = nullptr;
    }
    capacity = 0;
    shadow.clear();
}

size_t ShadowBuffer::upload(WGPUDevice device, WGPUQueue queue, WGPUBufferUsage usage, const void* dataIn,
                            size_t bytes, size_t elementSize, uint32_t& ranges) {
    if (bytes == 0) {
        return 0;
    }
    const auto* data = static_cast<const uint8_t*>(dataIn);
    if (bytes > capacity || buffer == nullptr) {
        release();
        capacity = (bytes + kCapacityStep - 1) / kCapacityStep * kCapacityStep;
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.label = label("xt_wgpu geometry");
        d.usage = usage | WGPUBufferUsage_CopyDst;
        d.size = capacity;
        buffer = wgpuDeviceCreateBuffer(device, &d);
        ++generation;
        if (buffer == nullptr) {
            capacity = 0;
            return 0;
        }
        wgpuQueueWriteBuffer(queue, buffer, 0, data, bytes);
        shadow.assign(data, data + bytes);
        ++ranges;
        return bytes;
    }
    // Element-wise diff against the shadow; runs closer than the merge gap are merged.
    const size_t gap = mergeGapBytes();
    size_t written = 0;
    const size_t common = std::min(bytes, shadow.size());
    const size_t commonElems = common / elementSize;
    size_t runStart = 0, runEnd = 0;
    bool open = false;
    auto flush = [&]() {
        if (!open) return;
        wgpuQueueWriteBuffer(queue, buffer, runStart, data + runStart, runEnd - runStart);
        ++ranges;
        written += runEnd - runStart;
        open = false;
    };
    for (size_t e = 0; e < commonElems; ++e) {
        const size_t off = e * elementSize;
        if (std::memcmp(data + off, shadow.data() + off, elementSize) == 0) {
            continue;
        }
        if (open && off - runEnd > gap) {
            flush();
        }
        if (!open) {
            runStart = off;
            open = true;
        }
        runEnd = off + elementSize;
    }
    const size_t tailStart = commonElems * elementSize;
    if (bytes > tailStart) {
        if (open && tailStart - runEnd > gap) {
            flush();
        }
        if (!open) {
            runStart = tailStart;
            open = true;
        }
        runEnd = bytes;
    }
    flush();
    shadow.assign(data, data + bytes);
    return written;
}

void Geometry::shutdown() {
    m_vertices.release();
    m_indices.release();
    m_instances.release();
    m_vertexBytes = m_indexBytes = m_instanceBytes = 0;
}

size_t Geometry::upload(WGPUDevice device, WGPUQueue queue, const gpu::GpuFrame& frame, uint32_t& ranges) {
    m_vertexBytes = frame.vertices.size() * sizeof(gpu::Vertex);
    m_indexBytes = frame.indices.size() * sizeof(uint32_t);
    m_instanceBytes = frame.particleInstances.size() * sizeof(gpu::ParticleInstance);
    size_t written = 0;
    written += m_vertices.upload(device, queue, WGPUBufferUsage_Vertex, frame.vertices.data(), m_vertexBytes,
                                 sizeof(gpu::Vertex), ranges);
    written += m_indices.upload(device, queue, WGPUBufferUsage_Index, frame.indices.data(), m_indexBytes,
                                sizeof(uint32_t), ranges);
    written += m_instances.upload(device, queue, WGPUBufferUsage_Vertex, frame.particleInstances.data(),
                                  m_instanceBytes, sizeof(gpu::ParticleInstance), ranges);
    return written;
}

}  // namespace xt::wgpu::detail
