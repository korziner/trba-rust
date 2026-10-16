#!/usr/bin/env bash
# ============================================================================
# run_cuda_smoke.sh — собрать CUDA-бэкенд и прогнать короткий smoke-прогон.
#
#   ./scripts/run_cuda_smoke.sh              # build + cargo test + smoke
#   SANITIZE=1 ./scripts/run_cuda_smoke.sh   # то же под /usr/local/cuda-13.4/bin/compute-sanitizer
#   DEBUG_SYNC=1 ./scripts/run_cuda_smoke.sh # назвать ядро, на котором падает
#   TESTS_ONLY=1 ./scripts/run_cuda_smoke.sh # только cargo test (секунды)
#   ARCH=86 BATCH=4 IMG_W=256 SMOKE=4 ./scripts/run_cuda_smoke.sh
#
# Датасет не нужен: --smoke генерирует случайные картинки и метки сам,
# charset.txt скрипт создаёт во временной папке.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

BATCH="${BATCH:-2}"          # маленький батч: следы каждого слоя живут в памяти до backward
IMG_H="${IMG_H:-32}"
IMG_W="${IMG_W:-128}"        # T = IMG_W/8 признаков для энкодера
MAX_LEN="${MAX_LEN:-24}"
SMOKE="${SMOKE:-2}"          # число батчей
PCIE="${PCIE:-0}"            # 1 = напечатать фактический объём трафика host<->device
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
ARCH="${ARCH:-${TRBA_CUDA_ARCH:-}}"
SANITIZE="${SANITIZE:-0}"
DEBUG_SYNC="${DEBUG_SYNC:-0}"   # 1 = синхронизация после каждого запуска ядра
TESTS_ONLY="${TESTS_ONLY:-0}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[1;32m✔\033[0m %s\n' "$*"; }
bad()  { printf '    \033[1;31m✘\033[0m %s\n' "$*"; }
die()  { bad "$*"; exit 1; }

# ---------------------------------------------------------------- 0. окружение
step "0. Окружение"
command -v cargo >/dev/null || die "нет cargo (curl https://sh.rustup.rs -sSf | sh -s -- -y)"

NVCC=""
for cand in "${TRBA_NVCC:-}" "${CUDA_PATH:+$CUDA_PATH/bin/nvcc}" /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc "$(command -v nvcc 2>/dev/null || true)"; do
  [ -n "$cand" ] && [ -x "$cand" ] && { NVCC="$cand"; break; }
done
[ -n "$NVCC" ] || die "nvcc не найден. Установите CUDA Toolkit или задайте TRBA_NVCC=/path/to/nvcc"
ok "nvcc: $NVCC ($($NVCC --version | awk '/release/{print $6}' | tr -d ','))"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
else
  bad "nvidia-smi не найден — драйвера NVIDIA нет, запуск упадёт на первом CUDA-вызове"
  CC=""
fi

# ARCH: явно > из драйвера > дефолт build.rs (fatbin 75/86/89)
if [ -z "$ARCH" ]; then ARCH="${CC:-}"; fi
if [ -n "$ARCH" ]; then
  export TRBA_CUDA_ARCH="$ARCH"
  ok "архитектура: sm_$ARCH"
else
  ok "архитектура: не задана, build.rs соберёт fatbin 75/86/89 + PTX"
fi

# TRBA_CUDA_LIB: если cudart_static лежит не рядом с nvcc (pip-пакеты, conda)
if [ -n "${TRBA_CUDA_LIB:-}" ]; then ok "TRBA_CUDA_LIB=$TRBA_CUDA_LIB"; fi

export TRBA_REQUIRE_CUDA=1   # build.rs и бинарник обязаны упасть, а не уйти на CPU

# ------------------------------------------------------------------- 1. сборка
step "1. Сборка CUDA-бэкенда (TRBA_REQUIRE_CUDA=1 — откат на CPU запрещён)"
cargo build --release -j "$JOBS" 2>&1 | tee "$WORK/build.log" | grep -E "cargo:warning|^error" | head -20
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || die "сборка не удалась (код $rc) — см. $WORK/build.log"
grep -q "CUDA backend built" "$WORK/build.log" || die "build.rs не собрал CUDA-бэкенд (см. $WORK/build.log)"
ok "слинкован CUDA-бэкенд: target/release/trba-rust"

# ------------------------------------------------------------ 2. юнит-тесты ядер
step "2. cargo test — каждое ядро сверяется с эталоном (секунды на GPU)"
cargo test --release -j "$JOBS" 2>&1 | tee "$WORK/test.log" | grep -E "^test |test result:|CUDA FATAL|panicked" | head -40
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  bad "тесты упали (код $rc) — см. $WORK/test.log"
  grep -E "CUDA FATAL|panicked|assertion" "$WORK/test.log" | head -10 | sed 's/^/      /'
  exit 1
fi
ok "тесты прошли: $(grep -oE '[0-9]+ passed' "$WORK/test.log" | head -1)"
[ "$TESTS_ONLY" = "1" ] && { ok "TESTS_ONLY=1 — пропускаю smoke"; exit 0; }

# ------------------------------------------------------------------ 3. smoke
step "3. Smoke-прогон: $SMOKE батча по $BATCH, ${IMG_H}x${IMG_W}, бэкенд cuda"
CHARSET="$WORK/charset.txt"
: > "$CHARSET"
printf '<PAD>\n<SOS>\n<EOS>\n' >> "$CHARSET"
for i in $(seq 4 60); do printf 't%s\n' "$i" >> "$CHARSET"; done
EXP="$WORK/exp"; mkdir -p "$EXP"

RUN=(./target/release/trba-rust --smoke "$SMOKE" --batch-size "$BATCH" \
     --img-h "$IMG_H" --img-w "$IMG_W" --max-len "$MAX_LEN" \
     --charset "$CHARSET" "$EXP")
if [ "$DEBUG_SYNC" = "1" ]; then
  # Без этого «illegal memory access» прилетает в случайном cudaFree/cudaMemcpy
  # через сотни запусков после виновника. Замедляет в несколько раз — годится,
  # чтобы найти падение, а не чтобы тренироваться.
  step "3a. TRBA_CUDA_DEBUG_SYNC=1 — ошибка будет названа по месту запуска ядра"
  export TRBA_CUDA_DEBUG_SYNC=1
fi

if [ "$SANITIZE" = "1" ]; then
  command -v /usr/local/cuda-13.4/bin/compute-sanitizer >/dev/null \
    || die "/usr/local/cuda-13.4/bin/compute-sanitizer не найден (входит в CUDA Toolkit: \$CUDA_PATH/bin//usr/local/cuda-13.4/bin/compute-sanitizer)"
  step "3b. /usr/local/cuda-13.4/bin/compute-sanitizer memcheck (медленно, зато ловит OOB и гонки)"
  RUN=(/usr/local/cuda-13.4/bin/compute-sanitizer --tool memcheck --error-exitcode 9 --launch-timeout 120 "${RUN[@]}")
fi

if [ "$PCIE" = "1" ]; then
  # Считает байты, прошедшие через cuda_memcpy_h2d/d2h, и печатает итог на выходе.
  # Прогоните с SMOKE=2 и SMOKE=4: разница, делённая на 2, — это трафик одного
  # батча без разовых расходов (загрузка весов, чекпойнт).
  export TRBA_PCIE_COUNTER=1
  step "3a. замер PCIe-трафика включён (TRBA_PCIE_COUNTER=1)"
fi

printf '    %s\n' "${RUN[*]}"
"${RUN[@]}" 2>&1 | tee "$WORK/smoke.log"
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  bad "smoke упал (код $rc) — см. $WORK/smoke.log"
  grep -E "CUDA FATAL|panicked|Error" "$WORK/smoke.log" | head -10 | sed 's/^/      /'
  exit 1
fi

grep -q "бэкенд: cuda" "$WORK/smoke.log" || die "бинарник запустился НЕ на CUDA-бэкенде"
ok "бэкенд: $(grep -o 'бэкенд: [a-z]* | .*' "$WORK/smoke.log" | head -1)"
grep -q "✅ smoke завершён" "$WORK/smoke.log" || die "smoke не дошёл до конца"
ok "smoke завершён"
grep -E "^Эпоха" "$WORK/smoke.log" | sed 's/^/    /'

# loss должен быть конечным: NaN/inf означает, что где-то в градиентах мусор
if grep -qiE "loss (nan|inf)" "$WORK/smoke.log"; then
  bad "loss не конечен — смотрите [CUDA FATAL]/DIAG выше"
  exit 1
fi
ok "loss конечен"

cat <<EOF

Готово. Логи: $WORK/{build,test,smoke}.log
Дальше:
  * полный тренинг:  ./target/release/trba-rust train.csv val.csv exp --batch-size 32 --epochs 100
  * отладка памяти:  SANITIZE=1 BATCH=1 IMG_W=64 SMOKE=1 ./scripts/run_cuda_smoke.sh
  * проверить логику ядер без GPU:  make -f tools/Makefile logic   (или g++ -O2 -std=c++17 tools/kernel_logic_check.cpp)
EOF
