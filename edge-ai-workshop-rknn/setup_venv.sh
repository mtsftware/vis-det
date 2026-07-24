#!/usr/bin/env bash
# ==============================================================================
# setup_venv.sh — RKNN Pipeline Python sanal ortam kurulum betiği
#
# rknn-toolkit-lite2 2.3.2  PyPI'dan otomatik indirilir (internete gerek var)
#
# Kullanım:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh               # PyPI'dan kur (tavsiye edilen)
#   ./setup_venv.sh wheel.whl     # yerel wheel (offline / farklı sürüm)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/.venv"

# PyPI'daki paket ismi ve sürümü
RKNN_PYPI_PACKAGE="rknn-toolkit-lite2==2.3.2"
# wheel dosyası arama ismi (eski/offline kullanım için fallback)
WHEEL_FALLBACK="rknn_toolkit_lite-1.7.5-cp38-cp38-linux_aarch64.whl"

# ── Renk / log ────────────────────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m';     NC='\033[0m'
step() { echo -e "\n${BLUE}▶${NC} ${BOLD}$*${NC}"; }
ok()   { echo -e "  ${GREEN}✔${NC}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
die()  { echo -e "\n${RED}${BOLD}✘  HATA: $*${NC}\n" >&2; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   RKNN Vision Pipeline — Python Sanal Ortam Kurulumu    ║"
echo "║   rknn-toolkit-lite2 2.3.2  •  Python 3.8  •  aarch64  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Dizin : ${SCRIPT_DIR}"
echo "  Venv  : ${VENV}"
echo

# ==============================================================================
# 1. Mimari — aarch64 zorunlu
# ==============================================================================
step "Mimari kontrol..."
[[ "$(uname -m)" == "aarch64" ]] \
    || die "Bu betik aarch64 boardlar içindir. Mevcut: $(uname -m)"
ok "aarch64  •  $(uname -r)"

# ==============================================================================
# 2. Python 3.8
# ==============================================================================
step "Python 3.8 aranıyor..."
PY38=""
for cmd in python3.8 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        _ver="$("$cmd" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
            2>/dev/null || true)"
        if [[ "$_ver" == "3.8" ]]; then
            PY38="$(command -v "$cmd")"
            break
        fi
    fi
done

if [[ -z "$PY38" ]]; then
    echo -e "\n${RED}Python 3.8 bulunamadı.${NC}\n"
    echo "  Ubuntu 20.04 : sudo apt install -y python3.8 python3.8-venv python3.8-dev"
    echo "  Ubuntu 22.04 : sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "                 sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev"
    die "Python 3.8 gerekli."
fi
"$PY38" -m venv --help &>/dev/null \
    || die "python3.8-venv eksik. Çözüm: sudo apt install python3.8-venv"
ok "$("$PY38" --version 2>&1)  →  ${PY38}"

# ==============================================================================
# 3. İnternet + RKNN kurulum kaynağı kararı
# ==============================================================================
step "RKNN kurulum kaynağı belirleniyor..."

LOCAL_WHEEL="${1:-}"   # CLI argümanı ile yerel wheel verilebilir
USE_PYPI=false

if [[ -n "$LOCAL_WHEEL" ]]; then
    # Argüman verilmiş — yerel wheel kullan
    [[ -f "$LOCAL_WHEEL" ]] || die "Belirtilen wheel bulunamadı: $LOCAL_WHEEL"
    ok "Yerel wheel kullanılacak: $LOCAL_WHEEL"
else
    # İnternet bağlantısı var mı?
    if curl -s --connect-timeout 5 https://pypi.org > /dev/null 2>&1; then
        USE_PYPI=true
        ok "İnternet bağlantısı OK — PyPI'dan kurulacak: ${RKNN_PYPI_PACKAGE}"
    else
        # PyPI yok — yerel wheel ara
        warn "İnternet bağlantısı yok. Yerel wheel aranıyor..."
        for _d in "${SCRIPT_DIR}" "$(pwd)" "${HOME}" "${HOME}/Downloads"; do
            if [[ -f "${_d}/${WHEEL_FALLBACK}" ]]; then
                LOCAL_WHEEL="${_d}/${WHEEL_FALLBACK}"
                break
            fi
        done
        if [[ -n "$LOCAL_WHEEL" ]]; then
            ok "Yerel wheel bulundu: $LOCAL_WHEEL"
        else
            die "İnternet yok ve yerel wheel de bulunamadı.\n" \
                "  Board'u internete bağlayın ya da:\n" \
                "  ./setup_venv.sh /yol/to/rknn_toolkit_lite2-2.3.2-cp38-...-aarch64.whl"
        fi
    fi
fi

# ==============================================================================
# 4. Venv oluştur
#    --system-site-packages: BSP'nin OpenCV/GStreamer gibi paketlerini devral
# ==============================================================================
step "Sanal ortam oluşturuluyor: ${VENV}"
"$PY38" -m venv --system-site-packages "${VENV}"

PIP="${VENV}/bin/pip"
PYTHON="${VENV}/bin/python"

ok "Python : $("$PYTHON" --version 2>&1)"

# ==============================================================================
# 5. pip güncelle
# ==============================================================================
step "pip güncelleniyor..."
"$PIP" install -q --upgrade pip setuptools wheel
ok "pip $(${PIP} --version | awk '{print $2}')"

# ==============================================================================
# 6. numpy — RKNN'den ÖNCE, sürüm kısıtı önemli
# ==============================================================================
step "numpy kuruluyor  (>=1.24, <2.0)..."
"$PIP" install -q "numpy>=1.24.0,<2.0.0"
ok "numpy $("$PYTHON" -c 'import numpy; print(numpy.__version__)')"

# ==============================================================================
# 7. RKNN Toolkit Lite 2 — PyPI veya yerel wheel
# ==============================================================================
step "RKNN Toolkit Lite 2 kuruluyor..."

if [[ "$USE_PYPI" == true ]]; then
    # PyPI'dan kur — bağımlılıkları da otomatik çeksin (ruamel.yaml dahil)
    "$PIP" install -q \
        --extra-index-url https://pypi.org/simple \
        "${RKNN_PYPI_PACKAGE}"
    ok "rknn-toolkit-lite2 2.3.2 PyPI'dan kuruldu"
else
    # Yerel wheel — bağımlılıkları elle kur
    "$PIP" install -q --no-deps --no-cache-dir "$LOCAL_WHEEL"
    # ruamel.yaml: rknn-toolkit-lite 1.x bağımlılığı
    "$PIP" install -q "ruamel.yaml"
    ok "rknn-toolkit-lite yerel wheel'den kuruldu + ruamel.yaml"
fi

# ==============================================================================
# 8. Diğer bağımlılıklar
# ==============================================================================
step "Diğer bağımlılıklar kuruluyor..."

REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    _TMPDIR="$(mktemp -d)"
    grep -v -i "rknn" "$REQ_FILE" > "${_TMPDIR}/req_filtered.txt"
    "$PIP" install -q -r "${_TMPDIR}/req_filtered.txt"
    rm -rf "$_TMPDIR"
    ok "requirements.txt uygulandı"
fi

# scipy ve psutil her zaman açıkça kur
"$PIP" install -q "scipy>=1.7.0,<2.0.0" "psutil>=5.9.0"
ok "scipy  $("$PYTHON" -c 'import scipy; print(scipy.__version__)')"
ok "psutil $("$PYTHON" -c 'import psutil; print(psutil.__version__)')"

# opencv — sistem sürümü yoksa headless kur
if ! "$PYTHON" -c "import cv2" &>/dev/null; then
    "$PIP" install -q "opencv-python-headless>=4.8.0,<5.0.0"
fi
ok "opencv $("$PYTHON" -c 'import cv2; print(cv2.__version__)')"

# flask
"$PIP" install -q "flask>=3.0.0"
ok "flask  $("$PYTHON" -c 'import flask; print(flask.__version__)' 2>/dev/null || echo 'OK')"

# ==============================================================================
# 9. librknnrt.so — LD_LIBRARY_PATH hook
# ==============================================================================
step "LD_LIBRARY_PATH hook ekleniyor..."
ACTIVATE="${VENV}/bin/activate"

if ! grep -q "librknnrt" "$ACTIVATE" 2>/dev/null; then
    cat >> "$ACTIVATE" << 'HOOK_EOF'

# ── RKNN runtime shared library (librknnrt.so) ────────────────────────────────
_rknn_lib_found=0
for _rknn_dir in /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /opt/rknn/lib; do
    if compgen -G "${_rknn_dir}/librknnrt.so*" > /dev/null 2>&1; then
        export LD_LIBRARY_PATH="${_rknn_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        _rknn_lib_found=1
        break
    fi
done
[ "$_rknn_lib_found" -eq 0 ] && echo "[RKNN WARN] librknnrt.so bulunamadı." >&2
unset _rknn_dir _rknn_lib_found
HOOK_EOF
    ok "LD_LIBRARY_PATH hook eklendi"
else
    ok "LD_LIBRARY_PATH hook zaten mevcut"
fi

# ==============================================================================
# 10. librknnrt.so varlık kontrolü
# ==============================================================================
step "librknnrt.so kontrol ediliyor..."
_SO_FOUND=""
for _p in /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /opt/rknn/lib; do
    _hit="$(ls "$_p"/librknnrt.so* 2>/dev/null | head -1 || true)"
    if [[ -n "$_hit" ]]; then _SO_FOUND="$_hit"; break; fi
done

if [[ -n "$_SO_FOUND" ]]; then
    ok "librknnrt.so : ${_SO_FOUND}"
    export LD_LIBRARY_PATH="$(dirname "$_SO_FOUND")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    warn "librknnrt.so bulunamadı — BSP kurulu değil olabilir"
    warn "RKNN çalışmayacak. sudo find / -name 'librknnrt.so*' 2>/dev/null"
fi

# ==============================================================================
# 11. Kurulum doğrulama
# ==============================================================================
step "Kurulum doğrulanıyor..."
echo

VERIFY="$("$PYTHON" << 'PYEOF' 2>&1
results = []
mods = [
    ("rknnlite.api", "RKNNLite", "rknnlite"),
    ("numpy",        "",         "numpy"),
    ("cv2",          "",         "opencv"),
    ("flask",        "",         "flask"),
    ("scipy",        "",         "scipy"),
    ("psutil",       "",         "psutil"),
]
for modname, attr, label in mods:
    try:
        m = __import__(modname, fromlist=[attr] if attr else [])
        if attr:
            getattr(m, attr)
        ver = getattr(m, "__version__", "OK")
        results.append(("ok", f"{label:<12} {ver}"))
    except ImportError as e:
        err = str(e)
        if "librknnrt" in err or (".so" in err and "rknn" in err.lower()):
            results.append(("warn", f"{label:<12} KURULU ama librknnrt.so eksik"))
        else:
            results.append(("err",  f"{label:<12} EKSIK — {err}"))
for s, m in results:
    print(f"{s}|{m}")
PYEOF
)"

FAIL=0
while IFS='|' read -r status msg; do
    # RKNN import sırasında gelen log satırlarını atla (| içermez)
    [[ "$status" =~ ^(ok|warn|err)$ ]] || continue
    if   [[ "$status" == "ok"   ]]; then echo -e "  ${GREEN}✔${NC}  $msg"
    elif [[ "$status" == "warn" ]]; then echo -e "  ${YELLOW}⚠${NC}  $msg"
    else                                  echo -e "  ${RED}✘${NC}  $msg"; FAIL=1
    fi
done <<< "$VERIFY"

# ==============================================================================
# 12. Sonuç
# ==============================================================================
echo
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  Kurulum başarıyla tamamlandı!${NC}"
else
    echo -e "${YELLOW}${BOLD}  Kurulum tamamlandı (bazı uyarılar var)${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
echo
echo -e "  ${BOLD}Aktive et ve çalıştır:${NC}"
echo -e "    ${YELLOW}source ${VENV}/bin/activate${NC}"
echo -e "    ${YELLOW}python main.py${NC}"
echo
echo -e "  ${BOLD}RKNN model:${NC}  ${SCRIPT_DIR}/yolov8n_int8.rknn"
echo
echo -e "  ${BOLD}Kurulu paketler:${NC}"
"$PIP" list 2>/dev/null \
    | grep -Ei "^(rknn|numpy|opencv|flask|scipy|psutil)" \
    | awk '{printf "    %-32s %s\n", $1, $2}'
echo
