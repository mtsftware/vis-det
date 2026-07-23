#include "buffer/DmaBufferPool.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// linux/dma-heap.h ve linux/dma-buf.h bazı minimal imajlarda yok — kernel
// UAPI'leri (Linux 5.6+ stabil) yerel tanımlanıyor.
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

struct dma_buf_sync_local {
    uint64_t flags;
};
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_IOCTL_SYNC_LOCAL _IOW('b', 0, struct dma_buf_sync_local)

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

struct DmaBufferPool::Impl {
    struct Slot {
        int fd = -1;
        void* virt = nullptr;
        bool in_use = false;
        bool cpu_read_open = false;
    };
    struct Pool {
        uint32_t buffer_size = 0;
        HeapKind heap = HeapKind::CachedDma32;
        std::vector<Slot> slots;
    };

    mutable std::mutex mtx;
    std::condition_variable cv;
    std::map<std::string, Pool> pools;
    bool shutting_down = false;

    int heap_fd_cached = -1;
    int heap_fd_uncached = -1;

    ~Impl() { destroyAllLocked(); }

    void destroyAllLocked() {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto& kv : pools) {
            for (auto& s : kv.second.slots) {
                if (s.in_use) {
                    std::cerr << "[DmaBufferPool] UYARI: '" << kv.first
                              << "' havuzunda iade edilmemiş slot\n";
                    continue;
                }
                if (s.virt) munmap(s.virt, kv.second.buffer_size);
                if (s.fd >= 0) close(s.fd);
            }
        }
        pools.clear();
        if (heap_fd_cached >= 0) close(heap_fd_cached);
        if (heap_fd_uncached >= 0) close(heap_fd_uncached);
        heap_fd_cached = heap_fd_uncached = -1;
    }

    int heapFdFor(HeapKind kind) {
        if (kind == HeapKind::CachedDma32) {
            if (heap_fd_cached < 0) {
                heap_fd_cached =
                    open("/dev/dma_heap/system-dma32", O_RDONLY | O_CLOEXEC);
                if (heap_fd_cached < 0) {
                    std::cerr << "[DmaBufferPool] UYARI: system-dma32 (cached) yok, "
                                 "uncached heap'e düşülüyor\n";
                    return heapFdFor(HeapKind::UncachedDma32);
                }
            }
            return heap_fd_cached;
        }
        if (heap_fd_uncached < 0) {
            heap_fd_uncached =
                open("/dev/dma_heap/system-uncached-dma32", O_RDONLY | O_CLOEXEC);
            if (heap_fd_uncached < 0) {
                std::cerr << "[DmaBufferPool] system-uncached-dma32 heap açılamadı\n";
            }
        }
        return heap_fd_uncached;
    }

    bool allocSlot(Pool& pool, Slot& slot) {
        int heap_fd = heapFdFor(pool.heap);
        if (heap_fd < 0) return false;

        dma_heap_allocation_data alloc_data{};
        alloc_data.len = pool.buffer_size;
        alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
            std::cerr << "[DmaBufferPool] DMA_HEAP_IOCTL_ALLOC başarısız (boyut="
                      << pool.buffer_size << ")\n";
            return false;
        }
        slot.fd = static_cast<int>(alloc_data.fd);
        slot.virt = mmap(nullptr, pool.buffer_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, slot.fd, 0);
        if (slot.virt == MAP_FAILED) {
            std::cerr << "[DmaBufferPool] mmap başarısız\n";
            close(slot.fd);
            slot.fd = -1;
            slot.virt = nullptr;
            return false;
        }
        return true;
    }

    void releaseSlot(const std::string& pool_name, size_t slot_index) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = pools.find(pool_name);
        if (it == pools.end() || slot_index >= it->second.slots.size()) return;
        Slot& s = it->second.slots[slot_index];

        if (s.cpu_read_open) {
            dma_buf_sync_local sync{};
            sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
            ioctl(s.fd, DMA_BUF_IOCTL_SYNC_LOCAL, &sync);
            s.cpu_read_open = false;
        }

        s.in_use = false;
        cv.notify_all();
    }
};

namespace {

// acquire/tryAcquireFor ortak gövdesi. weak_impl: deleter havuzdan uzun
// yaşayan bir buffer'da use-after-free yapmasın diye.
DmaBufferPtr acquireImpl(const std::shared_ptr<DmaBufferPool::Impl>& impl,
                         const std::string& name, int timeout_ms) {
    std::unique_lock<std::mutex> lk(impl->mtx);

    auto it = impl->pools.find(name);
    if (it == impl->pools.end()) {
        std::cerr << "[DmaBufferPool] acquire: '" << name << "' diye bir havuz yok\n";
        return nullptr;
    }

    auto find_free = [&]() -> int {
        auto& slots = impl->pools[name].slots;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i].in_use) return static_cast<int>(i);
        }
        return -1;
    };

    int idx = -1;
    auto pred = [&] {
        if (impl->shutting_down) return true;
        idx = find_free();
        return idx >= 0;
    };

    if (timeout_ms < 0) {
        impl->cv.wait(lk, pred);
    } else {
        if (!impl->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) {
            return nullptr;  // zaman aşımı — geri basınç sinyali
        }
    }
    if (impl->shutting_down || idx < 0) return nullptr;

    DmaBufferPool::Impl::Pool& pool = impl->pools[name];
    DmaBufferPool::Impl::Slot& slot = pool.slots[static_cast<size_t>(idx)];
    slot.in_use = true;

    auto buf = std::make_shared<DmaBuffer>();
    buf->fd = slot.fd;
    buf->virt_addr = slot.virt;
    buf->size = pool.buffer_size;

    std::weak_ptr<DmaBufferPool::Impl> weak_impl = impl;
    std::string pool_name = name;
    size_t slot_index = static_cast<size_t>(idx);
    buf->release_fn = [weak_impl, pool_name, slot_index]() {
        if (auto locked = weak_impl.lock()) {
            locked->releaseSlot(pool_name, slot_index);
        }
    };
    return buf;
}

}  // namespace

DmaBufferPool::DmaBufferPool() : impl_(std::make_shared<Impl>()) {}

DmaBufferPool::~DmaBufferPool() { shutdown(); }

bool DmaBufferPool::createPool(const std::string& name, uint32_t buffer_size,
                                uint32_t slot_count, HeapKind heap) {
    if (buffer_size == 0 || slot_count == 0) return false;

    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->shutting_down) return false;

    auto it = impl_->pools.find(name);
    if (it != impl_->pools.end()) {
        bool same = it->second.buffer_size == buffer_size &&
                    it->second.slots.size() == slot_count && it->second.heap == heap;
        if (!same) {
            std::cerr << "[DmaBufferPool] createPool: '" << name
                      << "' zaten farklı parametrelerle var\n";
        }
        return same;
    }

    Impl::Pool pool;
    pool.buffer_size = buffer_size;
    pool.heap = heap;
    pool.slots.resize(slot_count);
    for (auto& slot : pool.slots) {
        if (!impl_->allocSlot(pool, slot)) {
            for (auto& s : pool.slots) {
                if (s.virt) munmap(s.virt, buffer_size);
                if (s.fd >= 0) close(s.fd);
            }
            return false;
        }
    }
    impl_->pools.emplace(name, std::move(pool));
    return true;
}

bool DmaBufferPool::ensurePool(const std::string& name, uint32_t buffer_size,
                                uint32_t slot_count, HeapKind heap) {
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        auto it = impl_->pools.find(name);
        if (it != impl_->pools.end()) {
            return it->second.buffer_size == buffer_size;
        }
    }
    return createPool(name, buffer_size, slot_count, heap);
}

DmaBufferPtr DmaBufferPool::acquire(const std::string& name) {
    return acquireImpl(impl_, name, -1);
}

DmaBufferPtr DmaBufferPool::tryAcquireFor(const std::string& name, int timeout_ms) {
    return acquireImpl(impl_, name, timeout_ms < 0 ? 0 : timeout_ms);
}

void DmaBufferPool::syncCpuReadBegin(const DmaBufferPtr& buffer) {
    if (!buffer || buffer->fd < 0) return;

    std::lock_guard<std::mutex> lk(impl_->mtx);
    for (auto& kv : impl_->pools) {
        if (kv.second.heap != HeapKind::CachedDma32) continue;  // uncached: no-op
        for (auto& s : kv.second.slots) {
            if (s.fd == buffer->fd) {
                dma_buf_sync_local sync{};
                sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
                ioctl(s.fd, DMA_BUF_IOCTL_SYNC_LOCAL, &sync);
                s.cpu_read_open = true;
                return;
            }
        }
    }
}

void DmaBufferPool::shutdown() {
    {
        std::unique_lock<std::mutex> lk(impl_->mtx);
        if (impl_->shutting_down) return;
        impl_->shutting_down = true;
        impl_->cv.notify_all();  // bekleyen acquire'lar nullptr ile dönsün

        bool all_free = impl_->cv.wait_for(lk, std::chrono::seconds(2), [&] {
            for (auto& kv : impl_->pools) {
                for (auto& s : kv.second.slots) {
                    if (s.in_use) return false;
                }
            }
            return true;
        });
        if (!all_free) {
            std::cerr << "[DmaBufferPool] shutdown: 2s içinde iade edilmeyen "
                         "buffer(lar) var — ilgili fd'ler sızdırılacak\n";
        }
    }
    impl_->destroyAllLocked();
}

bool DmaBufferPool::getStats(const std::string& name, PoolStats& out) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->pools.find(name);
    if (it == impl_->pools.end()) return false;
    out.slot_count = static_cast<uint32_t>(it->second.slots.size());
    out.in_use = 0;
    for (auto& s : it->second.slots) {
        if (s.in_use) ++out.in_use;
    }
    return true;
}