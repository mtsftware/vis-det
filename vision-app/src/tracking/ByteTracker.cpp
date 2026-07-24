#include "tracking/ByteTracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <utility>
#include <vector>

// ============================================================================
// Kalman filtresi — referans: bytetracker_fix.py KalmanFilter (C++
// kalmanFilter.cpp'nin Python portu, burada tekrar C++'a cevrilmis).
// Durum: [cx, cy, a, h, vcx, vcy, va, vh] (a = aspect ratio = w/h).
// H (olcum matrisi) hep [I4 | 0] oldugu icin (ilk 4 durumu secer), genel
// matris carpimi yerine dogrudan alt-matris islemleri kullanilir — Eigen
// gibi harici bir lineer cebir kutuphanesi YOK (CLAUDE.md: paket kurulumu
// yasak), sabit boyutlu diziler + elle yazilmis carpim/ters-alma yeterli.
// ============================================================================
namespace kalman {

constexpr float kStdWeightPos = 1.0f / 20.0f;
constexpr float kStdWeightVel = 1.0f / 160.0f;

// [cx,cy,a,h] -> mean[8] (hiz=0), cov[8][8] (diag)
void initiate(const float measurement[4], float mean[8], float cov[8][8]) {
    for (int i = 0; i < 4; ++i) mean[i] = measurement[i];
    for (int i = 4; i < 8; ++i) mean[i] = 0.0f;

    float h = measurement[3];
    float std[8] = {
        2 * kStdWeightPos * h, 2 * kStdWeightPos * h, 1e-2f, 2 * kStdWeightPos * h,
        10 * kStdWeightVel * h, 10 * kStdWeightVel * h, 1e-5f, 10 * kStdWeightVel * h,
    };
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) cov[i][j] = (i == j) ? std[i] * std[i] : 0.0f;
}

// Zaman guncellemesi: mean_pred = F@mean, cov_pred = F@cov@F^T + Q.
// F = I8, F[i][4+i]=1 (i<4, dt=1) — yani F@X: satir i(<4) = X[i]+X[4+i].
void predict(float mean[8], float cov[8][8]) {
    float h = mean[3];
    float std_pos[4] = {kStdWeightPos * h, kStdWeightPos * h, 1e-2f, kStdWeightPos * h};
    float std_vel[4] = {kStdWeightVel * h, kStdWeightVel * h, 1e-5f, kStdWeightVel * h};

    float Q[8][8] = {};
    for (int i = 0; i < 4; ++i) Q[i][i] = std_pos[i] * std_pos[i];
    for (int i = 0; i < 4; ++i) Q[4 + i][4 + i] = std_vel[i] * std_vel[i];

    float mean_pred[8];
    for (int i = 0; i < 4; ++i) mean_pred[i] = mean[i] + mean[4 + i];
    for (int i = 4; i < 8; ++i) mean_pred[i] = mean[i];

    // FC = F @ cov: satir i(<4) = cov[i]+cov[4+i]; satir i(>=4) = cov[i]
    float FC[8][8];
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 4; ++i) FC[i][j] = cov[i][j] + cov[4 + i][j];
        for (int i = 4; i < 8; ++i) FC[i][j] = cov[i][j];
    }
    // cov_pred = FC @ F^T: sutun j(<4) = FC[:,j]+FC[:,4+j]; sutun j(>=4) = FC[:,j]
    float cov_pred[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) cov_pred[i][j] = FC[i][j] + FC[i][4 + j];
        for (int j = 4; j < 8; ++j) cov_pred[i][j] = FC[i][j];
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) cov_pred[i][j] += Q[i][j];

    for (int i = 0; i < 8; ++i) mean[i] = mean_pred[i];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) cov[i][j] = cov_pred[i][j];
}

// Olcum uzayina izdusum: H=[I4|0] oldugu icin mean_proj=mean[:4],
// cov_proj = cov[:4,:4] + R.
void project(const float mean[8], const float cov[8][8], float mean_proj[4],
             float cov_proj[4][4]) {
    float h = mean[3];
    float std[4] = {kStdWeightPos * h, kStdWeightPos * h, 1e-1f, kStdWeightPos * h};

    for (int i = 0; i < 4; ++i) mean_proj[i] = mean[i];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) cov_proj[i][j] = cov[i][j];
    for (int i = 0; i < 4; ++i) cov_proj[i][i] += std[i] * std[i];
}

// 4x4 Gauss-Jordan ters alma (kismi pivotlu). Basarisizsa false doner
// (dejenere matris — cagiran guncellemeyi atlar, cokme yerine).
bool invert4x4(const float m[4][4], float inv[4][4]) {
    float a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = m[i][j];
        for (int j = 0; j < 4; ++j) a[i][4 + j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        float best = std::fabs(a[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            if (std::fabs(a[r][col]) > best) {
                best = std::fabs(a[r][col]);
                piv = r;
            }
        }
        if (best < 1e-12f) return false;
        if (piv != col) {
            for (int j = 0; j < 8; ++j) std::swap(a[col][j], a[piv][j]);
        }
        float pivval = a[col][col];
        for (int j = 0; j < 8; ++j) a[col][j] /= pivval;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            float factor = a[r][col];
            if (factor == 0.0f) continue;
            for (int j = 0; j < 8; ++j) a[r][j] -= factor * a[col][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) inv[i][j] = a[i][4 + j];
    return true;
}

// Olcum guncellemesi: K = cov@H^T@inv(cov_proj) = cov[:, :4]@inv(cov_proj).
// new_mean = mean + K@(measurement-mean_proj); new_cov = cov - K@cov_proj@K^T.
void update(float mean[8], float cov[8][8], const float measurement[4]) {
    float mean_proj[4];
    float cov_proj[4][4];
    project(mean, cov, mean_proj, cov_proj);

    float cov_proj_inv[4][4];
    if (!invert4x4(cov_proj, cov_proj_inv)) {
        return;  // dejenere — guncellemeyi atla (mean/cov degismez)
    }

    // K[8][4] = cov[:, :4] @ cov_proj_inv
    float K[8][4];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += cov[i][k] * cov_proj_inv[k][j];
            K[i][j] = s;
        }
    }

    float innovation[4];
    for (int i = 0; i < 4; ++i) innovation[i] = measurement[i] - mean_proj[i];

    float new_mean[8];
    for (int i = 0; i < 8; ++i) {
        float s = mean[i];
        for (int j = 0; j < 4; ++j) s += K[i][j] * innovation[j];
        new_mean[i] = s;
    }

    // KC[8][4] = K @ cov_proj
    float KC[8][4];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += K[i][k] * cov_proj[k][j];
            KC[i][j] = s;
        }
    }
    // new_cov = cov - KC @ K^T
    float new_cov[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += KC[i][k] * K[j][k];
            new_cov[i][j] = cov[i][j] - s;
        }
    }

    for (int i = 0; i < 8; ++i) mean[i] = new_mean[i];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) cov[i][j] = new_cov[i][j];
}

}  // namespace kalman

// ============================================================================
// STrack — referans: bytetracker_fix.py STrack (C++ STrack.cpp'nin portu).
// ============================================================================
namespace {

int g_id_counter = 0;

struct STrack {
    float init_tlwh[4];  // ham dedektor kutusu: [x_tl, y_tl, w, h]
    float score;
    int class_id;

    bool is_activated = false;
    int track_id = 0;
    TrackState state = TrackState::New;

    bool has_kf = false;
    float mean[8] = {};
    float cov[8][8] = {};

    int frame_id = 0;
    int start_frame = 0;
    int tracklet_len = 0;

    float tlwh[4];  // mevcut tahmin (Kalman'dan turetilir, has_kf yoksa init_tlwh)

    STrack(const float tlwh_in[4], float score_, int class_id_)
        : score(score_), class_id(class_id_) {
        for (int i = 0; i < 4; ++i) {
            init_tlwh[i] = tlwh_in[i];
            tlwh[i] = tlwh_in[i];
        }
    }

    static void tlwhToXyah(const float t[4], float out[4]) {
        out[0] = t[0] + t[2] / 2.0f;
        out[1] = t[1] + t[3] / 2.0f;
        out[2] = t[2] / t[3];
        out[3] = t[3];
    }

    void updateTlwhFromMean() {
        if (!has_kf) return;
        float a = std::max(mean[2], 1e-6f);  // FIX (bytetracker_fix.py ile ayni): aspect sifir olmasin
        float h = std::max(mean[3], 1e-6f);  // FIX: yukseklik sifir olmasin
        float w = a * h;
        tlwh[2] = w;
        tlwh[3] = h;
        tlwh[0] = mean[0] - w / 2.0f;
        tlwh[1] = mean[1] - h / 2.0f;
    }

    void getTlbr(float out[4]) const {
        out[0] = tlwh[0];
        out[1] = tlwh[1];
        out[2] = tlwh[0] + tlwh[2];
        out[3] = tlwh[1] + tlwh[3];
    }

    void activate(int frame_id_) {
        track_id = ++g_id_counter;
        float xyah[4];
        tlwhToXyah(init_tlwh, xyah);
        kalman::initiate(xyah, mean, cov);
        has_kf = true;
        updateTlwhFromMean();

        tracklet_len = 0;
        state = TrackState::Tracked;
        frame_id = frame_id_;
        start_frame = frame_id_;
        if (frame_id_ == 1) is_activated = true;
    }

    void reActivate(const STrack& new_track, int frame_id_, bool new_id) {
        float xyah[4];
        tlwhToXyah(new_track.init_tlwh, xyah);
        kalman::update(mean, cov, xyah);
        updateTlwhFromMean();

        tracklet_len = 0;
        state = TrackState::Tracked;
        is_activated = true;
        frame_id = frame_id_;
        score = new_track.score;
        class_id = new_track.class_id;
        if (new_id) track_id = ++g_id_counter;
    }

    void updateMatched(const STrack& new_track, int frame_id_) {
        frame_id = frame_id_;
        tracklet_len++;

        float xyah[4];
        tlwhToXyah(new_track.init_tlwh, xyah);
        kalman::update(mean, cov, xyah);
        updateTlwhFromMean();

        state = TrackState::Tracked;
        is_activated = true;
        score = new_track.score;
        class_id = new_track.class_id;
    }

    void markLost() { state = TrackState::Lost; }
    void markRemoved() { state = TrackState::Removed; }
};

using STrackPtr = std::shared_ptr<STrack>;

void multiPredict(const std::vector<STrackPtr>& tracks) {
    for (auto& t : tracks) {
        if (t->state != TrackState::Tracked) t->mean[7] = 0.0f;  // kayip: dikey hiz sifirla
        kalman::predict(t->mean, t->cov);
        t->updateTlwhFromMean();
    }
}

// IoU-mesafe matrisi (1-IoU), union=0 durumunda NaN uretmeden 1.0 (max
// mesafe) doner — referans: bytetracker_fix.py _iou_distance FIX notu.
std::vector<std::vector<float>> iouDistance(const std::vector<STrack*>& a,
                                             const std::vector<STrack*>& b) {
    size_t M = a.size(), N = b.size();
    std::vector<std::vector<float>> dist(M, std::vector<float>(N, 0.0f));
    if (M == 0 || N == 0) return dist;

    for (size_t i = 0; i < M; ++i) {
        float atlbr[4];
        a[i]->getTlbr(atlbr);
        float areaA = std::max(0.0f, atlbr[2] - atlbr[0]) * std::max(0.0f, atlbr[3] - atlbr[1]);
        for (size_t j = 0; j < N; ++j) {
            float btlbr[4];
            b[j]->getTlbr(btlbr);
            float areaB =
                std::max(0.0f, btlbr[2] - btlbr[0]) * std::max(0.0f, btlbr[3] - btlbr[1]);

            float ix1 = std::max(atlbr[0], btlbr[0]);
            float iy1 = std::max(atlbr[1], btlbr[1]);
            float ix2 = std::min(atlbr[2], btlbr[2]);
            float iy2 = std::min(atlbr[3], btlbr[3]);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float uni = areaA + areaB - inter;

            float d = (uni > 0.0f) ? (1.0f - inter / uni) : 1.0f;
            if (!std::isfinite(d)) d = 1.0f;
            dist[i][j] = d;
        }
    }
    return dist;
}

// ----------------------------------------------------------------------------
// Macar algoritmasi (Kuhn-Munkres, O(n^2*m) potansiyel yontemi — e-maxx
// klasik implementasyonu). scipy.optimize.linear_sum_assignment'in C++
// karsiligi (harici bagimlilik yok). n<=m varsayar; n>m ise transpoze
// edilip cagrilir (linearAssignment sarmalayicisinda).
// ----------------------------------------------------------------------------
std::vector<int> hungarianAssign(const std::vector<std::vector<float>>& cost) {
    int n = static_cast<int>(cost.size());
    if (n == 0) return {};
    int m = static_cast<int>(cost[0].size());
    if (m == 0) return std::vector<int>(n, -1);

    const double INF = 1e18;
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<bool> used(m + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = -1;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                double cur = static_cast<double>(cost[i0 - 1][j - 1]) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] != 0) assignment[p[j] - 1] = j - 1;
    }
    return assignment;
}

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;  // (a_index, b_index)
    std::vector<int> unmatched_a;
    std::vector<int> unmatched_b;
};

// Esik degerli Macar algoritmasi — referans: bytetracker_fix.py
// _linear_assignment(). cost[i][j] <= thresh olmayan eslesmeler atilir
// (her iki taraf da "unmatched" listesine duser).
//
// n_a/n_b DISARIDAN acikca verilir — cost.size()/cost[0].size()'a
// GUVENILMEZ, cunku vector<vector<float>>(n_a=0, ...) 0 SATIRLI bir
// matris uretir ve bu durumda n_b (sutun sayisi) bilgisi TAMAMEN
// KAYBOLUR (cost[0] diye bir satir yok). BUG (bulunup duzeltildi):
// n_a=0 oldugunda (ornegin ilk eslestirmede strack_pool bos) eski kod
// m'yi de 0 sayiyordu, boylece TUM n_b adayi "unmatched_b" listesine
// eklenmek yerine sessizce kayboluyordu — sonucta hicbir yeni track
// ASLA olusturulamiyordu (Adim 6'ya hic aday gelmiyordu), dolayisiyla
// ekrana hicbir rectangle cizilmiyordu.
AssignmentResult linearAssignment(const std::vector<std::vector<float>>& cost, int n_a, int n_b,
                                   float thresh) {
    AssignmentResult result;
    int n = n_a;
    int m = n_b;

    if (n == 0 || m == 0) {
        for (int i = 0; i < n; ++i) result.unmatched_a.push_back(i);
        for (int j = 0; j < m; ++j) result.unmatched_b.push_back(j);
        return result;
    }

    std::vector<int> row_assign;
    if (n <= m) {
        row_assign = hungarianAssign(cost);
    } else {
        std::vector<std::vector<float>> cost_t(m, std::vector<float>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) cost_t[j][i] = cost[i][j];
        std::vector<int> col_assign = hungarianAssign(cost_t);  // size m, values in [0,n) or -1
        row_assign.assign(n, -1);
        for (int j = 0; j < m; ++j) {
            if (col_assign[j] != -1) row_assign[col_assign[j]] = j;
        }
    }

    std::vector<bool> used_b(m, false);
    for (int i = 0; i < n; ++i) {
        int j = row_assign[i];
        if (j >= 0 && cost[i][j] <= thresh) {
            result.matches.emplace_back(i, j);
            used_b[j] = true;
        } else {
            result.unmatched_a.push_back(i);
        }
    }
    for (int j = 0; j < m; ++j) {
        if (!used_b[j]) result.unmatched_b.push_back(j);
    }
    return result;
}

std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr>& a,
                                     const std::vector<STrackPtr>& b) {
    std::vector<STrackPtr> res = a;
    std::set<int> exists;
    for (auto& t : a) exists.insert(t->track_id);
    for (auto& t : b) {
        if (!exists.count(t->track_id)) {
            exists.insert(t->track_id);
            res.push_back(t);
        }
    }
    return res;
}

std::vector<STrackPtr> subStracks(const std::vector<STrackPtr>& a, const std::vector<STrackPtr>& b) {
    std::set<int> remove;
    for (auto& t : b) remove.insert(t->track_id);
    std::vector<STrackPtr> res;
    for (auto& t : a) {
        if (!remove.count(t->track_id)) res.push_back(t);
    }
    return res;
}

// Yuksek IoU'lu (< kDuplicateIouThresh mesafe) çift track'leri temizler;
// daha genc (kisa omurlu) olani kaldirir — referans: _remove_duplicates.
void removeDuplicates(std::vector<STrackPtr>& a, std::vector<STrackPtr>& b) {
    std::vector<STrack*> araw, braw;
    araw.reserve(a.size());
    braw.reserve(b.size());
    for (auto& t : a) araw.push_back(t.get());
    for (auto& t : b) braw.push_back(t.get());

    auto dist = iouDistance(araw, braw);
    std::set<size_t> dup_a, dup_b;
    for (size_t i = 0; i < araw.size(); ++i) {
        for (size_t j = 0; j < braw.size(); ++j) {
            if (dist[i][j] >= ByteTrackConfig::kDuplicateIouThresh) continue;
            int age_a = a[i]->frame_id - a[i]->start_frame;
            int age_b = b[j]->frame_id - b[j]->start_frame;
            if (age_a > age_b) {
                dup_b.insert(j);
            } else {
                dup_a.insert(i);
            }
        }
    }
    std::vector<STrackPtr> res_a, res_b;
    for (size_t i = 0; i < a.size(); ++i)
        if (!dup_a.count(i)) res_a.push_back(a[i]);
    for (size_t j = 0; j < b.size(); ++j)
        if (!dup_b.count(j)) res_b.push_back(b[j]);
    a = std::move(res_a);
    b = std::move(res_b);
}

}  // namespace

// ============================================================================
// ByteTracker — referans: bytetracker_fix.py BYTETracker.update() ile
// ADIM ADIM AYNI mantik (BYTETracker::update() C++ orijinalinin portu).
// ============================================================================
struct ByteTracker::Impl {
    int frame_id = 0;
    int max_time_lost =
        static_cast<int>(static_cast<float>(ByteTrackConfig::kFrameRate) / 30.0f *
                          ByteTrackConfig::kTrackBuffer);

    std::vector<STrackPtr> tracked_stracks;
    std::vector<STrackPtr> lost_stracks;
    std::vector<STrackPtr> removed_stracks;

    int ended_count = 0;  // markRemoved() ile sonlanan toplam track sayisi
};

ByteTracker::ByteTracker() : impl_(std::make_unique<Impl>()) {}
ByteTracker::~ByteTracker() = default;

std::vector<TrackedBox> ByteTracker::update(const std::vector<YoloDetection>& detections) {
    Impl& S = *impl_;
    S.frame_id++;
    const int frame_id = S.frame_id;

    std::vector<STrackPtr> activated_stracks;
    std::vector<STrackPtr> refind_stracks;
    std::vector<STrackPtr> lost_stracks_this_frame;
    std::vector<STrackPtr> removed_stracks_this_frame;

    // ── Adim 1: dedektorleri yuksek/dusuk guven olarak ayir ────────────────
    std::vector<STrackPtr> dets_high, dets_low;
    for (const auto& det : detections) {
        float tlwh[4] = {det.x, det.y, det.width, det.height};
        auto s = std::make_shared<STrack>(tlwh, det.confidence, det.class_id);
        if (det.confidence >= ByteTrackConfig::kTrackThresh) {
            dets_high.push_back(s);
        } else {
            dets_low.push_back(s);
        }
    }

    // ── Adim 2: mevcut track'leri ayir ──────────────────────────────────────
    std::vector<STrackPtr> unconfirmed, tracked_list;
    for (auto& t : S.tracked_stracks) {
        if (!t->is_activated) {
            unconfirmed.push_back(t);
        } else {
            tracked_list.push_back(t);
        }
    }

    // ── Adim 3: 1. eslestirme — yuksek guven det. x tum aktif ──────────────
    auto strack_pool = jointStracks(tracked_list, S.lost_stracks);
    multiPredict(strack_pool);

    std::vector<STrack*> pool_raw, high_raw;
    pool_raw.reserve(strack_pool.size());
    for (auto& t : strack_pool) pool_raw.push_back(t.get());
    high_raw.reserve(dets_high.size());
    for (auto& t : dets_high) high_raw.push_back(t.get());

    auto dist1 = iouDistance(pool_raw, high_raw);
    auto res1 = linearAssignment(dist1, static_cast<int>(pool_raw.size()),
                                  static_cast<int>(high_raw.size()), ByteTrackConfig::kMatchThresh);

    for (auto& mij : res1.matches) {
        auto& track = strack_pool[static_cast<size_t>(mij.first)];
        auto& det = dets_high[static_cast<size_t>(mij.second)];
        if (track->state == TrackState::Tracked) {
            track->updateMatched(*det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track->reActivate(*det, frame_id, false);
            refind_stracks.push_back(track);
        }
    }

    // ── Adim 4: 2. eslestirme — dusuk guven det. x eslesmemis Tracked ──────
    std::vector<STrackPtr> dets_cp;
    for (int j : res1.unmatched_b) dets_cp.push_back(dets_high[static_cast<size_t>(j)]);

    std::vector<STrackPtr> r_tracked;
    for (int i : res1.unmatched_a) {
        auto& t = strack_pool[static_cast<size_t>(i)];
        if (t->state == TrackState::Tracked) r_tracked.push_back(t);
    }

    std::vector<STrack*> r_tracked_raw, low_raw;
    r_tracked_raw.reserve(r_tracked.size());
    for (auto& t : r_tracked) r_tracked_raw.push_back(t.get());
    low_raw.reserve(dets_low.size());
    for (auto& t : dets_low) low_raw.push_back(t.get());

    auto dist2 = iouDistance(r_tracked_raw, low_raw);
    auto res2 = linearAssignment(dist2, static_cast<int>(r_tracked_raw.size()),
                                  static_cast<int>(low_raw.size()), ByteTrackConfig::kLowMatchThresh);

    for (auto& mij : res2.matches) {
        auto& track = r_tracked[static_cast<size_t>(mij.first)];
        auto& det = dets_low[static_cast<size_t>(mij.second)];
        if (track->state == TrackState::Tracked) {
            track->updateMatched(*det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track->reActivate(*det, frame_id, false);
            refind_stracks.push_back(track);
        }
    }
    for (int i : res2.unmatched_a) {
        auto& track = r_tracked[static_cast<size_t>(i)];
        if (track->state != TrackState::Lost) {
            track->markLost();
            lost_stracks_this_frame.push_back(track);
        }
    }

    // ── Adim 5: onaylanmamis track'ler x kalan yuksek guven det. ────────────
    std::vector<STrack*> unconfirmed_raw, dets_cp_raw;
    unconfirmed_raw.reserve(unconfirmed.size());
    for (auto& t : unconfirmed) unconfirmed_raw.push_back(t.get());
    dets_cp_raw.reserve(dets_cp.size());
    for (auto& t : dets_cp) dets_cp_raw.push_back(t.get());

    auto dist3 = iouDistance(unconfirmed_raw, dets_cp_raw);
    auto res3 = linearAssignment(dist3, static_cast<int>(unconfirmed_raw.size()),
                                  static_cast<int>(dets_cp_raw.size()),
                                  ByteTrackConfig::kUnconfirmedMatchThresh);

    for (auto& mij : res3.matches) {
        unconfirmed[static_cast<size_t>(mij.first)]->updateMatched(
            *dets_cp[static_cast<size_t>(mij.second)], frame_id);
        activated_stracks.push_back(unconfirmed[static_cast<size_t>(mij.first)]);
    }
    for (int i : res3.unmatched_a) {
        auto& t = unconfirmed[static_cast<size_t>(i)];
        t->markRemoved();
        removed_stracks_this_frame.push_back(t);
        S.ended_count++;
    }

    // ── Adim 6: eslesmeyen yuksek guven det. -> yeni track ──────────────────
    for (int i : res3.unmatched_b) {
        auto& track = dets_cp[static_cast<size_t>(i)];
        if (track->score < ByteTrackConfig::kHighThresh) continue;
        track->activate(frame_id);
        activated_stracks.push_back(track);
    }

    // ── Adim 7: max_time_lost'u gecen kayip track'leri sil ──────────────────
    for (auto& t : S.lost_stracks) {
        if (frame_id - t->frame_id > S.max_time_lost) {
            t->markRemoved();
            removed_stracks_this_frame.push_back(t);
            S.ended_count++;
        }
    }

    // ── Adim 8: track listelerini guncelle (Python ile BIREBIR AYNI sira) ──
    std::vector<STrackPtr> still_tracked;
    for (auto& t : S.tracked_stracks) {
        if (t->state == TrackState::Tracked) still_tracked.push_back(t);
    }
    S.tracked_stracks = jointStracks(still_tracked, activated_stracks);
    S.tracked_stracks = jointStracks(S.tracked_stracks, refind_stracks);

    S.lost_stracks = subStracks(S.lost_stracks, S.tracked_stracks);
    for (auto& t : lost_stracks_this_frame) S.lost_stracks.push_back(t);
    S.lost_stracks = subStracks(S.lost_stracks, S.removed_stracks);

    for (auto& t : removed_stracks_this_frame) S.removed_stracks.push_back(t);

    removeDuplicates(S.tracked_stracks, S.lost_stracks);

    // ── Cikti: SADECE Tracked + is_activated (kayip olan ANINDA duser) ─────
    std::vector<TrackedBox> output;
    output.reserve(S.tracked_stracks.size());
    for (auto& t : S.tracked_stracks) {
        if (!t->is_activated) continue;
        float tlbr[4];
        t->getTlbr(tlbr);
        TrackedBox tb;
        tb.track_id = t->track_id;
        tb.x = tlbr[0];
        tb.y = tlbr[1];
        tb.width = tlbr[2] - tlbr[0];
        tb.height = tlbr[3] - tlbr[1];
        tb.confidence = t->score;
        tb.class_id = t->class_id;
        output.push_back(tb);
    }

    // TESHIS: "hic cizilmiyor" arastirmasi icin — ByteTrack'in 2-kareli
    // onay mekanizmasinin (kUnconfirmedMatchThresh=0.7, Adim 5) track'leri
    // hic onaylamadan silip silmedigini gormek. unconfirmed_after >
    // unconfirmed(bu karenin basindaki) ise yeni track'ler onaylanmiyor
    // demektir — o zaman esik cok siki (detection kutulari kare-kare
    // yeterince ortusmuyor) olabilir.
    if (frame_id % 30 == 0) {
        size_t unconfirmed_after = 0;
        for (auto& t : S.tracked_stracks) {
            if (!t->is_activated) ++unconfirmed_after;
        }
        std::cout << "[ByteTracker] frame " << frame_id
                  << ": dets_high=" << dets_high.size() << " dets_low=" << dets_low.size()
                  << " stage1_match=" << res1.matches.size()
                  << " stage2_match=" << res2.matches.size()
                  << " stage3_match(onay)=" << res3.matches.size()
                  << " onay_bekleyen_bu_kare=" << unconfirmed.size()
                  << " onay_bekleyen_sonraki_kare=" << unconfirmed_after
                  << " tracked_stracks=" << S.tracked_stracks.size()
                  << " lost_stracks=" << S.lost_stracks.size()
                  << " CIKTI(cizilecek)=" << output.size() << "\n";
    }

    return output;
}

int ByteTracker::activeCount() const {
    int count = 0;
    for (auto& t : impl_->tracked_stracks) {
        if (t->is_activated) ++count;
    }
    return count;
}

int ByteTracker::totalTracked() const { return g_id_counter; }

int ByteTracker::lostCount() const { return impl_->ended_count; }
