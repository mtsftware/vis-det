"""
bytetracker_fix.py — BYTETracker Python portu (NXP Edge AI Workshop).

C++ orijinalinden (BYTETracker.cpp / STrack.cpp / kalmanFilter.cpp) bire bir çevrilmiştir.

Referans: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
          Zhang et al., ECCV 2022  |  https://github.com/ifzhang/ByteTrack

Kullanım:
    from bytetracker_fix import BYTETracker

    tracker = BYTETracker(frame_rate=30, track_buffer=30,
                          track_thresh=0.3, high_thresh=0.5, match_thresh=0.8)

    # Her frame'de:
    tracks = tracker.update(detections)
    # detections: [{'bbox':[x1,y1,x2,y2], 'label_id':int, 'confidence':float}, ...]
    # tracks:     [..., + 'track_id':int]

Düzeltmeler (bytetracker.py → bytetracker_fix.py):
    1. _iou_distance: union=0 durumunda NaN üretimi engellendi (np.where + nan_to_num)
    2. _linear_assignment: cost matrisinde NaN/inf varsa nan_to_num ile temizlenir
    3. STrack._update_tlwh: aspect ratio ve yükseklik sıfır olmasın diye guard eklendi
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────

class TrackState(IntEnum):
    New     = 0
    Tracked = 1
    Lost    = 2
    Removed = 3


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filtresi  (kalmanFilter.cpp → Python)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    8-boyutlu Kalman filtresi: sabit-hız hareket modeli.

    Durum vektörü  : [cx, cy, a, h,  vcx, vcy, va, vh]
    Ölçüm vektörü : [cx, cy, a, h]

    cx, cy = merkez koordinatları
    a      = en-boy oranı  (aspect ratio = w/h)
    h      = yükseklik

    C++ karşılığı: KalmanFilter::KalmanFilter() / predict() / project() / update()
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # F: hareket matrisi (8×8) — üst sağ blok dt ile dolu
        self.F = np.eye(8, dtype=np.float32)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # H: ölçüm matrisi (4×8) — ilk 4 durumu seçer
        self.H = np.eye(4, 8, dtype=np.float32)

        self._std_weight_pos = 1.0 / 20.0   # C++: _std_weight_position
        self._std_weight_vel = 1.0 / 160.0  # C++: _std_weight_velocity

    # ------------------------------------------------------------------ init

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        İlk ölçümden track başlat.
        C++ karşılığı: KalmanFilter::initiate()
        """
        mean = np.concatenate([measurement, np.zeros(4, dtype=np.float32)])

        h = measurement[3]
        std = np.array([
            2 * self._std_weight_pos * h,
            2 * self._std_weight_pos * h,
            1e-2,
            2 * self._std_weight_pos * h,
            10 * self._std_weight_vel * h,
            10 * self._std_weight_vel * h,
            1e-5,
            10 * self._std_weight_vel * h,
        ], dtype=np.float32)
        covariance = np.diag(std ** 2)
        return mean, covariance

    # --------------------------------------------------------------- predict

    def predict(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Zaman güncellemesi (prediction step).
        C++ karşılığı: KalmanFilter::predict()
        """
        h = mean[3]
        std_pos = np.array([
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-2,
            self._std_weight_pos * h,
        ], dtype=np.float32)
        std_vel = np.array([
            self._std_weight_vel * h,
            self._std_weight_vel * h,
            1e-5,
            self._std_weight_vel * h,
        ], dtype=np.float32)
        Q = np.diag(np.concatenate([std_pos, std_vel]) ** 2)

        mean_pred = self.F @ mean
        cov_pred  = self.F @ covariance @ self.F.T + Q
        return mean_pred, cov_pred

    # ---------------------------------------------------------------- project

    def project(self, mean: np.ndarray, covariance: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Durumu ölçüm uzayına yansıt.
        C++ karşılığı: KalmanFilter::project()
        """
        h = mean[3]
        std = np.array([
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-1,
            self._std_weight_pos * h,
        ], dtype=np.float32)
        R = np.diag(std ** 2)

        mean_proj = self.H @ mean
        cov_proj  = self.H @ covariance @ self.H.T + R
        return mean_proj, cov_proj

    # ----------------------------------------------------------------- update

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ölçüm güncellemesi (update step).
        C++ karşılığı: KalmanFilter::update()
        """
        mean_proj, cov_proj = self.project(mean, covariance)

        # Kalman kazancı: K = P·H^T·(H·P·H^T + R)^{-1}
        K          = covariance @ self.H.T @ np.linalg.inv(cov_proj)
        innovation = measurement - mean_proj
        new_mean   = mean + K @ innovation
        new_cov    = covariance - K @ cov_proj @ K.T
        return new_mean, new_cov

    # ---------------------------------------------------------- multi_predict

    def multi_predict(self, stracks: list[STrack]) -> None:
        """
        Birden fazla track için toplu prediction.
        C++ karşılığı: STrack::multi_predict()
        """
        for t in stracks:
            if t.state != TrackState.Tracked:
                # Kayıp track'ler için dikey hızı sıfırla
                t.mean[7] = 0.0
            t.mean, t.covariance = self.predict(t.mean, t.covariance)
            t._update_tlwh()


# ─────────────────────────────────────────────────────────────────────────────
# STrack  (STrack.cpp → Python)
# ─────────────────────────────────────────────────────────────────────────────

class STrack:
    """
    Tek bir nesne track'i.
    C++ karşılığı: STrack sınıfının tamamı.
    """

    _id_counter: int = 0

    def __init__(self, tlwh: list[float], score: float, label_id: int = 0) -> None:
        # Orijinal (ham dedektör) bounding box: [x_tl, y_tl, w, h]
        self._init_tlwh = np.array(tlwh, dtype=np.float32)

        self.score    = score
        self.label_id = label_id

        # Track durumu
        self.is_activated: bool       = False
        self.track_id:     int        = 0
        self.state:        TrackState = TrackState.New

        # Kalman durumu
        self.mean:       np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        # Zaman bilgisi
        self.frame_id:     int = 0
        self.start_frame:  int = 0
        self.tracklet_len: int = 0

        # Kalman filtresi referansı (activate() sırasında atanır)
        self._kf: KalmanFilter | None = None

        # Mevcut tlwh (Kalman'dan türetilir)
        self._tlwh: np.ndarray = self._init_tlwh.copy()

    # ────────────────────────────────────── sınıf metotları

    @classmethod
    def next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def total_created(cls) -> int:
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    # ────────────────────────────────────── koordinat dönüşümleri

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """[x_tl, y_tl, w, h] → [cx, cy, a, h]  (a = w/h)"""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[0] += ret[2] / 2.0  # cx
        ret[1] += ret[3] / 2.0  # cy
        ret[2] /= ret[3]        # aspect ratio
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """[x1, y1, x2, y2] → [x_tl, y_tl, w, h]"""
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret

    def _update_tlwh(self) -> None:
        """
        Kalman ortalamasından tlwh'yi yeniden hesapla.

        FIX: Kalman diverge ederse aspect ratio veya yükseklik sıfıra düşebilir;
             bunu 1e-6 ile sabitleriz, yoksa tlbr alan hesabında NaN üretilir.
        """
        if self.mean is None:
            self._tlwh = self._init_tlwh.copy()
            return
        ret = self.mean[:4].copy()
        ret[2] = max(float(ret[2]), 1e-6)   # FIX: aspect ratio sıfır olmasın
        ret[3] = max(float(ret[3]), 1e-6)   # FIX: yükseklik sıfır olmasın
        ret[2] *= ret[3]          # a*h = w
        ret[0] -= ret[2] / 2.0   # cx - w/2
        ret[1] -= ret[3] / 2.0   # cy - h/2
        self._tlwh = ret

    @property
    def tlwh(self) -> np.ndarray:
        """Mevcut [x_tl, y_tl, w, h] tahmini."""
        return self._tlwh.copy()

    @property
    def tlbr(self) -> np.ndarray:
        """Mevcut [x1, y1, x2, y2] tahmini."""
        ret = self._tlwh.copy()
        ret[2] += ret[0]
        ret[3] += ret[1]
        return ret

    @property
    def end_frame(self) -> int:
        return self.frame_id

    # ────────────────────────────────────── yaşam döngüsü

    def activate(self, kf: KalmanFilter, frame_id: int) -> None:
        """
        Yeni track'i başlat.
        C++ karşılığı: STrack::activate()
        """
        self._kf      = kf
        self.track_id = self.next_id()

        xyah = self.tlwh_to_xyah(self._init_tlwh)
        self.mean, self.covariance = kf.initiate(xyah)
        self._update_tlwh()

        self.tracklet_len = 0
        self.state        = TrackState.Tracked
        self.frame_id     = frame_id
        self.start_frame  = frame_id

        # C++ davranışı: yalnızca frame_id==1'de hemen aktive et
        if frame_id == 1:
            self.is_activated = True

    def re_activate(self, new_track: STrack, frame_id: int,
                    new_id: bool = False) -> None:
        """
        Kayıp track'i yeniden aktive et.
        C++ karşılığı: STrack::re_activate()
        """
        xyah = self.tlwh_to_xyah(new_track.tlwh)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, xyah)
        self._update_tlwh()

        self.tracklet_len = 0
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.score        = new_track.score
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: STrack, frame_id: int) -> None:
        """
        Aktif track'i güncelle.
        C++ karşılığı: STrack::update()
        """
        self.frame_id     = frame_id
        self.tracklet_len += 1

        xyah = self.tlwh_to_xyah(new_track.tlwh)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, xyah)
        self._update_tlwh()

        self.state        = TrackState.Tracked
        self.is_activated = True
        self.score        = new_track.score

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    def __repr__(self) -> str:
        return f'Track_{self.track_id}({self.start_frame}-{self.end_frame})'


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _iou_distance(a_tracks: list[STrack], b_tracks: list[STrack]) -> np.ndarray:
    """
    İki track listesi arasındaki IoU mesafe matrisi (1 - IoU).

    FIX: union=0 (degenerate bbox) durumunda inter/union → NaN üretiliyordu.
         np.where ile sıfır bölme engellendi; kalan NaN/inf → nan_to_num ile 1.0'a sabitlendi.

    C++ karşılığı: BYTETracker::iou_distance()
    """
    if not a_tracks or not b_tracks:
        return np.zeros((len(a_tracks), len(b_tracks)), dtype=np.float32)

    a_tlbr = np.array([t.tlbr for t in a_tracks], dtype=np.float32)  # (M,4)
    b_tlbr = np.array([t.tlbr for t in b_tracks], dtype=np.float32)  # (N,4)

    area_a = np.maximum(0, a_tlbr[:, 2] - a_tlbr[:, 0]) * np.maximum(0, a_tlbr[:, 3] - a_tlbr[:, 1])
    area_b = np.maximum(0, b_tlbr[:, 2] - b_tlbr[:, 0]) * np.maximum(0, b_tlbr[:, 3] - b_tlbr[:, 1])

    ix1 = np.maximum(a_tlbr[:, None, 0], b_tlbr[None, :, 0])
    iy1 = np.maximum(a_tlbr[:, None, 1], b_tlbr[None, :, 1])
    ix2 = np.minimum(a_tlbr[:, None, 2], b_tlbr[None, :, 2])
    iy2 = np.minimum(a_tlbr[:, None, 3], b_tlbr[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter

    # FIX: union=0 → IoU=0 (kesişim yok sayılır), NaN/inf üretilmez
    iou  = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - iou

    # FIX: Kalan olası NaN/inf değerlerini maksimum mesafeye (1.0) sabitle
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)

    return dist.astype(np.float32)


def _linear_assignment(cost: np.ndarray, thresh: float
                        ) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Eşik değerli Macar algoritması.

    FIX: cost matrisinde NaN/inf varsa scipy.linear_sum_assignment
         "matrix contains invalid numeric entries" hatası verir.
         Bu değerler nan_to_num ile 1.0'a (max maliyet) dönüştürülür.

    C++ karşılığı: BYTETracker::linear_assignment() (lapjv yerine scipy)
    """
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(cost.shape[0])),
                list(range(cost.shape[1])))

    # FIX: geçersiz sayısal değerleri temizle
    if not np.isfinite(cost).all():
        cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches     = []
    unmatched_a = set(range(cost.shape[0]))
    unmatched_b = set(range(cost.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append([r, c])
            unmatched_a.discard(r)
            unmatched_b.discard(c)

    return np.array(matches, dtype=int), list(unmatched_a), list(unmatched_b)


def _joint_stracks(lista: list[STrack], listb: list[STrack]) -> list[STrack]:
    """İki listeden tekrar olmadan birleşim oluştur."""
    exists = {t.track_id for t in lista}
    res = list(lista)
    for t in listb:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def _sub_stracks(lista: list[STrack], listb: list[STrack]) -> list[STrack]:
    """lista - listb (track_id'ye göre)."""
    remove = {t.track_id for t in listb}
    return [t for t in lista if t.track_id not in remove]


def _remove_duplicates(stracksa: list[STrack], stracksb: list[STrack]
                        ) -> tuple[list[STrack], list[STrack]]:
    """
    Yüksek IoU'lu çift track'leri temizle; daha genç olanı kaldır.
    C++ karşılığı: BYTETracker::remove_duplicate_stracks()
    """
    dist = _iou_distance(stracksa, stracksb)
    pairs = np.where(dist < 0.15)
    dupa: set[int] = set()
    dupb: set[int] = set()
    for p, q in zip(*pairs):
        age_a = stracksa[p].frame_id - stracksa[p].start_frame
        age_b = stracksb[q].frame_id - stracksb[q].start_frame
        if age_a > age_b:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


# ─────────────────────────────────────────────────────────────────────────────
# BYTETracker  (BYTETracker.cpp → Python)
# ─────────────────────────────────────────────────────────────────────────────

class BYTETracker:
    """
    BYTETracker: çift aşamalı IoU eşleştirmeli çok nesne takipçisi.

    C++ karşılığı: BYTETracker sınıfının tamamı.

    Parametreler (C++ constructor ile birebir):
        frame_rate   : Video kare hızı (default 30)
        track_buffer : Kayıp track'ler bu kadar frame sonra silinir (default 30)
        track_thresh : Yüksek güven eşiği (default 0.5)
        high_thresh  : Yeni track başlatma eşiği (default 0.6)
        match_thresh : IoU eşleştirme eşiği (default 0.8)
    """

    def __init__(self,
                 frame_rate:   int   = 30,
                 track_buffer: int   = 30,
                 track_thresh: float = 0.5,
                 high_thresh:  float = 0.6,
                 match_thresh: float = 0.8) -> None:
        self.track_thresh  = track_thresh
        self.high_thresh   = high_thresh
        self.match_thresh  = match_thresh
        self.frame_id      = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.kalman_filter    = KalmanFilter()
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self._track_lifespans: list[int] = []

    # ────────────────────────────────────────────────────────────── update()

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Bir frame'deki dedektörlerle tracker'ı güncelle.

        C++ karşılığı: BYTETracker::update()

        Args:
            detections: Dedektörden gelen sonuçlar.
                        Her eleman: {
                            'bbox':       [x1, y1, x2, y2],
                            'label_id':   int,
                            'confidence': float
                        }

        Returns:
            Aktif track listesi.
            Her eleman: {
                'bbox':       [x1, y1, x2, y2],   # Kalman tahminli, piksel
                'label_id':   int,
                'confidence': float,
                'track_id':   int                  # YENİ
            }
        """
        self.frame_id += 1

        activated_stracks: list[STrack] = []
        refind_stracks:    list[STrack] = []
        lost_stracks:      list[STrack] = []
        removed_stracks:   list[STrack] = []

        # ── Adım 1: Dedektörleri yüksek / düşük güven olarak ayır ──────────
        dets_high: list[STrack] = []
        dets_low:  list[STrack] = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            tlwh     = [x1, y1, x2 - x1, y2 - y1]
            score    = float(det['confidence'])
            label_id = int(det.get('label_id', 0))
            strack   = STrack(tlwh, score, label_id)
            if score >= self.track_thresh:
                dets_high.append(strack)
            else:
                dets_low.append(strack)

        # ── Adım 2: Mevcut track'leri ayır ──────────────────────────────────
        unconfirmed:     list[STrack] = []
        tracked_stracks: list[STrack] = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked_stracks.append(t)

        # ── Adım 3: 1. eşleştirme — yüksek güven det. × tüm aktif ──────────
        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)
        self.kalman_filter.multi_predict(strack_pool)

        dist1 = _iou_distance(strack_pool, dets_high)
        matches1, u_track1, u_det1 = _linear_assignment(dist1, self.match_thresh)

        for m in matches1:
            track = strack_pool[m[0]]
            det   = dets_high[m[1]]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ── Adım 4: 2. eşleştirme — düşük güven det. × eşleşmemiş ─────────
        dets_cp   = [dets_high[i] for i in u_det1]
        r_tracked = [strack_pool[i] for i in u_track1
                     if strack_pool[i].state == TrackState.Tracked]

        dist2 = _iou_distance(r_tracked, dets_low)
        matches2, u_track2, _ = _linear_assignment(dist2, 0.5)

        for m in matches2:
            track = r_tracked[m[0]]
            det   = dets_low[m[1]]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for i in u_track2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── Adım 5: Onaylanmamış track'ler × kalan yüksek güven det. ────────
        dist3 = _iou_distance(unconfirmed, dets_cp)
        matches3, u_unconf, u_det_cp = _linear_assignment(dist3, 0.7)

        for m in matches3:
            unconfirmed[m[0]].update(dets_cp[m[1]], self.frame_id)
            activated_stracks.append(unconfirmed[m[0]])

        for i in u_unconf:
            t = unconfirmed[i]
            self._track_lifespans.append(self.frame_id - t.start_frame)
            t.mark_removed()
            removed_stracks.append(t)

        # ── Adım 6: Eşleşmeyen yüksek güven det. → yeni track ──────────────
        for i in u_det_cp:
            track = dets_cp[i]
            if track.score < self.high_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # ── Adım 7: max_time_lost'u geçen kayıp track'leri sil ─────────────
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                self._track_lifespans.append(t.end_frame - t.start_frame)
                t.mark_removed()
                removed_stracks.append(t)

        # ── Adım 8: Track listelerini güncelle ──────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = _remove_duplicates(
            self.tracked_stracks, self.lost_stracks
        )

        # ── Çıktı: yalnızca aktive edilmiş track'ler ─────────────────────────
        output: list[dict] = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                continue
            tlbr = t.tlbr
            output.append({
                'bbox':       [int(tlbr[0]), int(tlbr[1]),
                               int(tlbr[2]), int(tlbr[3])],
                'label_id':   t.label_id,
                'confidence': float(t.score),
                'track_id':   t.track_id,
            })

        return output

    # ─────────────────────────────────────────────────────── istatistikler

    @property
    def track_lifespans(self) -> list[int]:
        """Sonlanan track'lerin ömürleri (frame sayısı). Track loss proxy."""
        return self._track_lifespans

    def summary(self) -> str:
        total_created = STrack.total_created()
        ended         = len(self._track_lifespans)
        alive         = total_created - ended
        avg_life      = (sum(self._track_lifespans) / ended) if ended else 0.0
        return (
            f"Track özeti: toplam={total_created} | "
            f"sonlanan={ended} | aktif={alive} | "
            f"ort_ömür={avg_life:.1f}f"
        )
