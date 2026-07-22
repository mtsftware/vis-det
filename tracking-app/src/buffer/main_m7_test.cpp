// M7 doğrulama: DmaBufferPool'un (04-IBufferPool.md implementasyonu) kart
// üzerinde davranış testi. RGA/NPU GEREKTİRMEZ — sadece /dev/dma_heap
// (yani kartta koşmalı, PC'de değil). Kapsam:
//
//   T1  createPool + idempotentlik (aynı parametre=true, farklı=false)
//   T2  Slot tükenince tryAcquireFor zaman aşımı (geri basınç)
//   T3  shared_ptr bırakınca slotun havuza OTOMATİK dönmesi (RAII)
//   T4  Bloklamalı acquire'ın, başka thread iade edince uyanması
//   T5  mmap üzerinden veri bütünlüğü (yaz-oku deseni)
//   T6  Çok-thread'li stres: 3 üretici x 300 tur, 4 slotluk havuz
//   T7  Uncached heap varyantı (kart destekliyorsa)
//   T8  shutdown'ın dolaşımdaki buffer iadesini beklemesi
//
// RgaPreprocessor'ın havuza geçişinin uçtan uca doğrulaması AYRICA
// m6_tracking_test'in yeniden koşulmasıyla yapılır (davranış birebir aynı
// kalmalı: 30 FPS decode, ~0.999 confidence, RGA hatasız).
//
// Derleme (kartta):
//   g++ -std=c++14 -O2 \
//     src/buffer/DmaBufferPool.cpp src/buffer/main_m7_test.cpp \
//     -Iinclude -lpthread -o m7_pool_test
//
// Çalıştırma:
//   ./m7_pool_test

#include "buffer/DmaBufferPool.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* what) {
    if (cond) {
        ++g_pass;
        std::cout << "[M7]   OK   " << what << "\n";
    } else {
        ++g_fail;
        std::cout << "[M7]   FAIL " << what << "\n";
    }
}

uint64_t nowMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

}  // namespace

int main() {
    std::cout << "[M7] DmaBufferPool testi başlıyor...\n";

    // ---- T1: createPool + idempotentlik ----
    {
        DmaBufferPool pool;
        check(pool.createPool("t1", 4096, 3, HeapKind::CachedDma32),
              "T1 createPool başarılı");
        check(pool.createPool("t1", 4096, 3, HeapKind::CachedDma32),
              "T1 aynı parametreyle tekrar createPool -> true (idempotent)");
        check(!pool.createPool("t1", 8192, 3, HeapKind::CachedDma32),
              "T1 farklı boyutla createPool -> false (sessiz boyutlandırma yok)");
        check(!pool.createPool("t1-bad", 0, 3, HeapKind::CachedDma32),
              "T1 sıfır boyut reddedilir");
        DmaBufferPool::PoolStats st;
        check(pool.getStats("t1", st) && st.slot_count == 3 && st.in_use == 0,
              "T1 stats: 3 slot, 0 kullanımda");
    }

    // ---- T2: geri basınç (tükenme -> zaman aşımı) ----
    {
        DmaBufferPool pool;
        pool.createPool("t2", 4096, 2, HeapKind::CachedDma32);
        DmaBufferPtr a = pool.acquire("t2");
        DmaBufferPtr b = pool.acquire("t2");
        check(a && b, "T2 2/2 slot alındı");

        uint64_t t0 = nowMs();
        DmaBufferPtr c = pool.tryAcquireFor("t2", 150);
        uint64_t elapsed = nowMs() - t0;
        check(c == nullptr, "T2 3. istek nullptr (havuz dolu)");
        check(elapsed >= 130, "T2 zaman aşımı gerçekten beklendi (>=130ms)");

        check(pool.tryAcquireFor("boyle-havuz-yok", 10) == nullptr,
              "T2 bilinmeyen havuz adı -> nullptr");
    }

    // ---- T3: RAII iadesi ----
    {
        DmaBufferPool pool;
        pool.createPool("t3", 4096, 1, HeapKind::CachedDma32);
        {
            DmaBufferPtr a = pool.acquire("t3");
            DmaBufferPool::PoolStats st;
            pool.getStats("t3", st);
            check(a && st.in_use == 1, "T3 alınınca in_use=1");
        }  // a burada düşer -> slot havuza dönmeli
        DmaBufferPool::PoolStats st;
        pool.getStats("t3", st);
        check(st.in_use == 0, "T3 shared_ptr düşünce in_use=0 (otomatik iade)");

        DmaBufferPtr b = pool.tryAcquireFor("t3", 50);
        check(b != nullptr, "T3 iade edilen slot yeniden alınabildi");

        // shared_ptr kopyası: son referans düşene kadar iade YOK.
        DmaBufferPtr b2 = b;
        b.reset();
        pool.getStats("t3", st);
        check(st.in_use == 1, "T3 kopya yaşarken slot havuza dönmez (refcount)");
        b2.reset();
        pool.getStats("t3", st);
        check(st.in_use == 0, "T3 son kopya düşünce döner");
    }

    // ---- T4: bloklamalı acquire başka thread'in iadesiyle uyanır ----
    {
        DmaBufferPool pool;
        pool.createPool("t4", 4096, 1, HeapKind::CachedDma32);
        DmaBufferPtr held = pool.acquire("t4");

        std::atomic<bool> got{false};
        std::atomic<uint64_t> wake_ms{0};
        uint64_t t0 = nowMs();
        std::thread waiter([&] {
            DmaBufferPtr b = pool.acquire("t4");  // bloklamalı
            wake_ms.store(nowMs() - t0);
            got.store(b != nullptr);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        held.reset();  // iade -> waiter uyanmalı
        waiter.join();
        check(got.load(), "T4 bloklanan acquire iadeyle buffer aldı");
        check(wake_ms.load() >= 100, "T4 gerçekten bekledi (>=100ms)");
    }

    // ---- T5: veri bütünlüğü ----
    {
        DmaBufferPool pool;
        pool.createPool("t5", 8192, 2, HeapKind::CachedDma32);
        DmaBufferPtr a = pool.acquire("t5");
        bool ok = a && a->virt_addr && a->size == 8192;
        if (ok) {
            uint8_t* p = static_cast<uint8_t*>(a->virt_addr);
            for (uint32_t i = 0; i < a->size; ++i) p[i] = static_cast<uint8_t>(i * 31u);
            for (uint32_t i = 0; i < a->size && ok; ++i) {
                ok = (p[i] == static_cast<uint8_t>(i * 31u));
            }
        }
        check(ok, "T5 mmap yaz-oku deseni birebir");
    }

    // ---- T6: çok-thread'li stres ----
    {
        DmaBufferPool pool;
        pool.createPool("t6", 16384, 4, HeapKind::CachedDma32);

        constexpr int kThreads = 3;
        constexpr int kIters = 300;
        std::atomic<int> success{0};
        std::atomic<int> corrupt{0};

        std::vector<std::thread> workers;
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([&, t] {
                for (int i = 0; i < kIters; ++i) {
                    DmaBufferPtr b = pool.acquire("t6");
                    if (!b || !b->virt_addr) continue;
                    uint8_t pattern = static_cast<uint8_t>((t * 97 + i) & 0xFF);
                    uint8_t* p = static_cast<uint8_t*>(b->virt_addr);
                    std::memset(p, pattern, b->size);
                    // Slot bizdeyken kimse üstüne yazamamalı:
                    bool ok = true;
                    for (uint32_t k = 0; k < b->size; k += 257) {
                        if (p[k] != pattern) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        success.fetch_add(1);
                    } else {
                        corrupt.fetch_add(1);
                    }
                }  // b düşer -> iade
            });
        }
        for (auto& w : workers) w.join();

        check(success.load() == kThreads * kIters,
              "T6 stres: tüm turlar başarılı (3x300)");
        check(corrupt.load() == 0, "T6 stres: hiç veri çakışması yok");
        DmaBufferPool::PoolStats st;
        pool.getStats("t6", st);
        check(st.in_use == 0, "T6 stres sonrası tüm slotlar iade edilmiş");
    }

    // ---- T7: uncached heap (varsa) ----
    {
        DmaBufferPool pool;
        if (pool.createPool("t7", 4096, 1, HeapKind::UncachedDma32)) {
            DmaBufferPtr a = pool.tryAcquireFor("t7", 100);
            bool ok = a && a->virt_addr;
            if (ok) {
                uint8_t* p = static_cast<uint8_t*>(a->virt_addr);
                p[0] = 0xAB;
                p[4095] = 0xCD;
                ok = (p[0] == 0xAB && p[4095] == 0xCD);
            }
            check(ok, "T7 uncached havuz alloc+yaz-oku");
        } else {
            std::cout << "[M7]   SKIP T7 uncached heap bu imajda yok\n";
        }
    }

    // ---- T8: shutdown dolaşımdaki iadeyi bekler ----
    {
        DmaBufferPool pool;
        pool.createPool("t8", 4096, 1, HeapKind::CachedDma32);
        DmaBufferPtr held = pool.acquire("t8");

        std::atomic<uint64_t> shutdown_ms{0};
        uint64_t t0 = nowMs();
        std::thread closer([&] {
            pool.shutdown();
            shutdown_ms.store(nowMs() - t0);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        held.reset();  // iade -> shutdown tamamlanabilmeli
        closer.join();
        check(shutdown_ms.load() >= 130,
              "T8 shutdown dolaşımdaki buffer'ı bekledi (>=130ms)");
    }

    std::cout << "\n[M7] SONUÇ: " << g_pass << " OK, " << g_fail << " FAIL\n";
    return g_fail == 0 ? 0 : 1;
}
