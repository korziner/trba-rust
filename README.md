# trba-rust
TRBA OCR port to Rust and CUDA + MoPE arch

```
OMP_NUM_THREADS=4 bash scripts/mope_menu.sh

── Что делаем?
   1) импорт датасета YeniseiGovReports-HWR в формат trba-rust (секунды)
   2) замер скорости на 500 строках (~1 мин, чтобы не запускать вслепую)
   3) обучить эксперта / продолжить обучение
   4) связать двух экспертов в MoPE и прогнать инференс
   5) корзины по длине: много экспертов (длины 4…40) и один франкен
   выбор [1-5, Enter = 1]: 5

▶ геометрия: --img-h 32 --img-w 160 --max-len 20 --hidden 64 --feat-dim 128

── Схема корзин?
   1) auto — смежные диапазоны, в каждом не меньше MIN_ROWS строк: корзин меньше, каждая плотная, паддинг минимален (рекомендуется)
   2) exact — ровно по одной длине 4,5,6,…,40: до 37 экспертов, длинные корзины жидкие или пустые
   3) cum — накопительно: корзина L видит все строки длиной ≤ L, последняя обучается дольше всех
   4) только план (ничего не обучать): показать корзины и их размеры
   выбор [1-4, Enter = 1]: 

── Минимум строк в корзине?
   1) 300 — корзин больше, эксперты слабее
   2) 1000 — равновесие
   3) 3000 — корзин меньше, эксперты сильнее
   4) 5000
   выбор [1-4, Enter = 1]: 4

── Эпох на корзину? (каждая корзина мала, поэтому эпох нужно больше, чем на полном датасете)
   1) 3 — быстро прикинуть
   2) 6
   3) 10 — рабочее качество
   4) 33 — максимум
   выбор [1-4, Enter = 1]: 4
   train: 124768 строк → 109570 годных
   val: 22401 строк → 20586 годных
🔤 charset YeniseiGovReports-HWR/charset_trba.txt: 130 токенов (3 служебных + 127 символов по train)
💾 YeniseiGovReports-HWR/train/trba_train.csv: 109570 строк
💾 YeniseiGovReports-HWR/val/trba_val.csv: 20568 строк
📏 длина текста: p50 5 · p90 11 · p99 15 · max 32
🖼  картинки (200 шт): ширина min 12 / p50 61 / max 198, высота min 8 / p50 19 / max 37
   препроцессинг trba-rust: RGB → resize с сохранением пропорций → белый холст, поэтому кадр 32×160 подходит и словам 30×22, и 140×29
🧹 отброшено: короче 2 17011, нет картинки 2, символ вне словаря 18
📄 YeniseiGovReports-HWR/trba_import_report.txt
🔤 словарь собран: 130138 строк из YeniseiGovReports-HWR/train/full_train.csv,YeniseiGovReports-HWR/val/full_val.csv, 127 символов + 3 служебных → YeniseiGovReports-HWR/charset_full.txt (пропущено заголовков 0, битых строк 0)
корпус: 109570 строк | длина min 2 · p50 5 · p90 11 · p99 15 · max 32
гистограмма длин: 2:18864 3:12693 4:13111 5:11258 6:10261 7:9699 8:8537 9:7948 10:5849 11:4287 12:2687 13:1694 14:994 15:753 16:391 17:207 18:144 19:78 20:32 21:34 22:14 23:18 24:6 25:6 26:3 32:2
⚠️  --hi 40 больше максимальной длины 32: корзины длиннее 32 будут пустыми и пропущены
корзина         длины   строк контроль  img_w  max_len
bucket_04_04        4…4   13111     2561     64        4
bucket_05_05        5…5   11258     2070     96        5
bucket_06_06        6…6   10261     1985     96        6
bucket_07_07        7…7    9699     1839    128        7
bucket_08_08        8…8    8537     1512    128        8
bucket_09_09        9…9    7948     1517    160        9
bucket_10_10      10…10    5849     1177    160       10
bucket_11_12      11…12    6974     1354    192       12
bucket_13_40      13…40    4376      832    256       40

▶ корзин: 9 | план: /home/korziner/Downloads/7c99..ZIP/trba-rust/YeniseiGovReports-HWR/train/buckets_plan.tsv
  у каждой корзины свой --max-len (= hi) и свой --img-w: паддинг не
  считается впустую, а CER каждой корзины измеряется на строках её длины.

▶ корзин 9 | потолок прогона max_len=40 | fd=128 эпох 88 batch 48

════ корзина bucket_04_04: длины 4…4, строк 13111, img 32×64, max_len 4 ════
total 34M
12M Oct 18 02:25 checkpoint_epoch024_loss2.0275.bin
12M Oct 18 05:53 checkpoint_epoch033_loss2.0205.bin
12M Oct 18 05:53 checkpoint_last.bin
🚀 TRBA (Rust + INT8 QAT) | бэкенд: cuda | NVIDIA CMP 50HX
📖 charset: 130 токенов | img 32x64 | max_len 4 | hidden 64 | слоёв BiLSTM 2
📄 train: 13111 (пропущено 0) | val: 2561 (пропущено 0)
⚙️  оптимизатор: Muon (momentum-free) для матриц, lr·sqrt(max_dim) + Adam для векторов, cap 10 | глобальный clip: первые 500 шаг(ов)
🧠 веса свёрток (2.63M) с градиентами и Adam-состоянием живут на GPU: ~40 MB VRAM, зато ~0.6 MB трафика через PCIe на батч вместо ~362 MB
   мало видеопамяти? TRBA_HOST_CONV_ADAM=1 вернёт оптимизатор на хост (+362 MB/батч по PCIe); TRBA_PCIE_COUNTER=1 печатает фактический трафик
📂 resume: exp_len_04_04/checkpoint_last.bin
   чекпоинт: 32×64 max_len 4 hidden 64 классов 130 rnn 2 feat 128
   запуск    : 32×64 max_len 4 hidden 64 классов 130 rnn 2 feat 128
   завершено эпох: 33 | продолжать с эпохи 34, батч 1 | шаг оптимизатора 22146 | Adam-моменты: восстановлены
💾 rolling-чекпоинт exp_len_04_04/checkpoint_last.bin: каждые 500 батч(ей) и в конце каждой эпохи (продолжить — тот же запуск без --no-resume)
[trba] int8 GEMM: WMMA m16n16k16 s8*s8->s32 (тензорные ядра, IMMA) | cc 7.5 | SM 56 | выбран автоматически
Эпоха 34/88 | 61s | loss 2.0370 (attn 2.0114 ctc 2.8645, w_ctc 0.030) | lr 3.53e-4
Эпоха 35/88 | 57s | loss 2.0287 (attn 2.0029 ctc 2.8627, w_ctc 0.030) | lr 3.45e-4
...
Эпоха 87/88 | 55s | loss 1.9238 (attn 1.8953 ctc 2.8440, w_ctc 0.030) | lr 6.51e-7
Эпоха 88/88 | 55s | loss 1.9225 (attn 1.8940 ctc 2.8441, w_ctc 0.030) | lr 1.63e-7
💾 exp_len_04_04/checkpoint_epoch088_loss1.9225.bin
🚀 TRBA (Rust + INT8 QAT) | бэкенд: cuda | NVIDIA CMP 50HX
📖 charset: 130 токенов | img 32x64 | max_len 4 | hidden 64 | слоёв BiLSTM 2
📖 exp_len_04_04/checkpoint_last.bin: эпоха 88
   exp_len_04_04 → перенесено 138 тензоров манифеста, пропущено 0
💾 experts/yen_04_04.mope (int8 MOPEBI01): 2.96 МБ вместо 11.75 МБ fp32, 2938484 параметров, 194 тензоров
   exp_len_04_04 [trba-rust-bin] img 32×64 max_len 4 hidden 64 feat 128 классов 130 rnn 2 (pad 0 sos 1 eos 2)

════ корзина bucket_05_05: длины 5…5, строк 11258, img 32×96, max_len 5 ════
total 46M
12M Oct 17 22:11 checkpoint_epoch010_loss2.7509.bin
12M Oct 18 02:35 checkpoint_epoch024_loss2.5829.bin
12M Oct 18 06:02 checkpoint_epoch033_loss2.5600.bin
12M Oct 18 06:02 checkpoint_last.bin
🚀 TRBA (Rust + INT8 QAT) | бэкенд: cuda | NVIDIA CMP 50HX
📖 charset: 130 токенов | img 32x96 | max_len 5 | hidden 64 | слоёв BiLSTM 2
📄 train: 11258 (пропущено 0) | val: 2070 (пропущено 0)
⚙️  оптимизатор: Muon (momentum-free) для матриц, lr·sqrt(max_dim) + Adam для векторов, cap 10 | глобальный clip: первые 500 шаг(ов)
🧠 веса свёрток (2.63M) с градиентами и Adam-состоянием живут на GPU: ~40 MB VRAM, зато ~0.6 MB трафика через PCIe на батч вместо ~362 MB
   мало видеопамяти? TRBA_HOST_CONV_ADAM=1 вернёт оптимизатор на хост (+362 MB/батч по PCIe); TRBA_PCIE_COUNTER=1 печатает фактический трафик
📂 resume: exp_len_05_05/checkpoint_last.bin
   чекпоинт: 32×96 max_len 5 hidden 64 классов 130 rnn 2 feat 128
   запуск    : 32×96 max_len 5 hidden 64 классов 130 rnn 2 feat 128
   завершено эпох: 33 | продолжать с эпохи 34, батч 1 | шаг оптимизатора 15573 | Adam-моменты: восстановлены
💾 rolling-чекпоинт exp_len_05_05/checkpoint_last.bin: каждые 500 батч(ей) и в конце каждой эпохи (продолжить — тот же запуск без --no-resume)
[trba] int8 GEMM: WMMA m16n16k16 s8*s8->s32 (тензорные ядра, IMMA) | cc 7.5 | SM 56 | выбран автоматически
Эпоха 34/88 | 67s | loss 2.7363 (attn 2.7245 ctc 3.1150, w_ctc 0.030) | lr 3.53e-4
Эпоха 35/88 | 83s | loss 2.6841 (attn 2.6705 ctc 3.1230, w_ctc 0.030) | lr 3.45e-4
...
════ корзина bucket_09_09: длины 9…9, строк 7948, img 32×160, max_len 9 ════
total 34M
12M Oct 18 00:05 checkpoint_epoch010_loss2.9355.bin
12M Oct 18 03:44 checkpoint_epoch024_loss2.6911.bin
12M Oct 18 03:44 checkpoint_last.bin
🚀 TRBA (Rust + INT8 QAT) | бэкенд: cuda | NVIDIA CMP 50HX
📖 charset: 130 токенов | img 32x160 | max_len 9 | hidden 64 | слоёв BiLSTM 2
📄 train: 7948 (пропущено 0) | val: 1517 (пропущено 0)
⚙️  оптимизатор: Muon (momentum-free) для матриц, lr·sqrt(max_dim) + Adam для векторов, cap 10 | глобальный clip: первые 500 шаг(ов)
🧠 веса свёрток (2.63M) с градиентами и Adam-состоянием живут на GPU: ~40 MB VRAM, зато ~0.6 MB трафика через PCIe на батч вместо ~362 MB
   мало видеопамяти? TRBA_HOST_CONV_ADAM=1 вернёт оптимизатор на хост (+362 MB/батч по PCIe); TRBA_PCIE_COUNTER=1 печатает фактический трафик
📂 resume: exp_len_09_09/checkpoint_last.bin
   чекпоинт: 32×160 max_len 9 hidden 64 классов 130 rnn 2 feat 128
   запуск    : 32×160 max_len 9 hidden 64 классов 130 rnn 2 feat 128
   завершено эпох: 24 | продолжать с эпохи 25, батч 1 | шаг оптимизатора 11928 | Adam-моменты: восстановлены
💾 rolling-чекпоинт exp_len_09_09/checkpoint_last.bin: каждые 500 батч(ей) и в конце каждой эпохи (продолжить — тот же запуск без --no-resume)
[trba] int8 GEMM: WMMA m16n16k16 s8*s8->s32 (тензорные ядра, IMMA) | cc 7.5 | SM 56 | выбран автоматически
Эпоха 25/88 | 93s | loss 2.7539 (attn 2.7442 ctc 3.0677, w_ctc 0.030) | lr 4.23e-4
Эпоха 26/88 | 91s | loss 2.7154 (attn 2.7046 ctc 3.0657, w_ctc 0.030) | lr 4.16e-4
...
Эпоха 68/88 | 94s | loss 2.4527 (attn 2.4340 ctc 3.0584, w_ctc 0.030) | lr 6.85e-5
Эпоха 69/88 | 93s | loss 2.4520 (attn 2.4333 ctc 3.0564, w_ctc 0.030) | lr 6.24e-5
```

## Backends

`build.rs` compiles one of two native backends with the same `extern "C"` surface:

* `cuda/kernels.cu` through `nvcc` — used whenever a CUDA toolkit is found
  (`TRBA_NVCC`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`, `PATH`);
* `cuda/cpu_backend.cpp` through the C++ compiler — fallback, or `TRBA_FORCE_CPU=1`.

Everything is linked statically, so no `LD_LIBRARY_PATH` is needed at run time.

**Always build the GPU binary with `TRBA_REQUIRE_CUDA=1`.** Without it a failed `nvcc`
run silently links the CPU backend and the binary still starts — you would train on the
CPU believing it is the GPU. With it, the build fails loudly, and `src/main.rs` also
refuses to start if the linked backend is not `cuda`.

## Building and running on a GPU

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader   # e.g. 7.5 -> sm_75

TRBA_REQUIRE_CUDA=1 TRBA_CUDA_ARCH=75 cargo build --release     # your card's sm_XX
cargo test --release                                            # 23 kernel tests, seconds on a GPU
./target/release/trba-rust train.csv val.csv exp --batch-size 32 --epochs 100
```

The first line printed must be `🚀 TRBA ... | бэкенд: cuda | NVIDIA ...`.

One command for the whole bring-up (build + kernel tests + a short smoke run on synthetic
data, no dataset required; `SANITIZE=1` additionally runs it under `compute-sanitizer`):

```bash
ARCH=75 ./scripts/run_cuda_smoke.sh
```

Smoke run by hand (needs a `charset.txt` whose first three tokens are `<PAD> <SOS> <EOS>`):

```bash
./target/release/trba-rust --smoke 2 --batch-size 2 --img-h 32 --img-w 128 \
    --max-len 24 --charset charset.txt exp_smoke
```

Start small: `NetTrace` keeps the pre/post-BN activations of all 28 conv layers alive until
the backward pass, and the device optimiser state below adds ~643 MiB, so `--batch-size 32
--img-w 1024` is a lot of device memory.

## PCIe traffic (this is what hurts on PCIe 2.0 x4)

Everything that used to travel per batch now stays on the device:

* the 2x2 maxpool runs on the GPU over int8, and its backward **recomputes the argmax** from the
  activations the trace already keeps — the u64 index tensor no longer exists (that was 256 MB
  per batch upstream, plus 256 MB of host RAM);
* `grad_input` reads the int8 fake-quant weights the forward GEMM already uploaded, so the fp32
  master weights (160.8 MB per batch) never cross the bus;
* the conv optimiser lives on the device: `w`, `g`, `m`, `v` of all 28 layers (42.15M
  parameters), the int32 `grad_W` accumulator folded into the fp32 gradient by a kernel, the
  global grad-norm clip assembled from one 8-byte device scalar, and QAT re-quantisation done on
  the GPU with only the 4-byte scale mirrored back per layer.  The arithmetic matches
  `src/opt.rs` (f64 intermediates, f32 state) and is covered by a bit-exact test.

Measured with `TRBA_PCIE_COUNTER=1` (prints the H2D/D2H totals at exit; run `--smoke 2` and
`--smoke 4`, divide the difference by two to get one batch, which removes the one-off costs):

| mode | MB per batch |
|------|--------------|
| host Adam + fp32 `grad_input` (the original design) | 361.4 |
| host Adam + int8 `grad_input` | 201.1 |
| device Adam + device re-quantisation (default) | **0.6** |

That weight traffic is independent of batch size and image size.  At `--batch-size 32 --img-h 32
--img-w 1024` add the activations that genuinely have to travel (3 MB of images in, 16 MB of
features to the host RNN and back): **~19 MB per batch instead of ~797 MB**, i.e. 15-19 ms
instead of 0.6-0.8 s on a pageable PCIe 2.0 x4 link.

| variable | effect |
|----------|--------|
| `TRBA_PCIE_COUNTER=1` | print the actual host<->device byte totals at exit (both backends) |
| `TRBA_HOST_CONV_ADAM=1` | keep the conv optimiser on the host: frees ~643 MiB of VRAM, costs +201 MB per batch over PCIe |
| `TRBA_GRAD_INPUT_FP32=1` | compute `grad_input` from the fp32 master weights (numerical reference): +161 MB per batch |
| `TRBA_GPU_RNN=1` | run the BiLSTM on the GPU instead of the host (stage 1: LSTM directions on device, projection on host) — see *BiLSTM on the GPU* |
| `TRBA_OPT=adam` | use the previous Adam for **all** parameters instead of Muon (A/B switch) — see *Muon instead of Adam* |
| `TRBA_MUON_Q=1` | MuonQ: quantise the gradient to int8 in blocks before the Muon step (regularisation, not speed — see below) |
| `TRBA_MUON_Q_BLOCK=N` | MuonQ block size, default 128 |
| `TRBA_MUON_CAP=X` | cap on `sqrt(min_dim)/\|\|g\|\|_F`, default 10 |
| `TRBA_MUON_AUTOSCALE=0` | disable the `lr * sqrt(max_dim)` rescaling |
| `TRBA_CLIP_WARMUP=N` / `--clip-warmup N` | apply the global grad-norm clip only on the first N steps, default 500 |

`PCIE=1 ARCH=75 ./scripts/run_cuda_smoke.sh` runs the smoke with the counter on.

Note that the CPU fallback backend is compiled with OpenMP; if your compiler rejects `-fopenmp`
the build prints a loud warning, because that silently makes every reference kernel
single-threaded.

## Picking batch size and crop width (`--autotune`)

`--autotune` searches the (batch_size, img_w) space on *your* GPU and reports throughput,
VRAM and PCIe cost per batch, then recommends a pair. Optuna-style: a budget of trials,
a deterministic sampler seeded by `--seed`, and an objective you choose.

```bash
# вся сетка (мелкие конфигурации первыми)
./target/release/trba-rust train.csv val.csv exp --charset charset.txt --autotune

# бюджет 24 пробы по 6 батчей, цель — картинки/с на GiB занятой памяти, предел 8 GB
./target/release/trba-rust train.csv val.csv exp --charset charset.txt \
    --autotune --autotune-trials 24 --autotune-batches 6 \
    --autotune-objective efficiency --autotune-vram 8000

# свои списки (например, только ширины из вашего пайплайна)
./target/release/trba-rust train.csv val.csv exp --charset charset.txt \
    --autotune --autotune-widths 256,384,512 --autotune-batch-sizes 4,8,16
```

Each trial is a **separate child process** running `--smoke K` with those shapes. That is
deliberate: `an illegal memory access was encountered` and CUDA OOM destroy the context, so
a crashing configuration cannot be measured from inside the same process — the parent just
records the failure (with the signal / stderr reason) and keeps searching. The report is
written to `<exp_dir>/autotune.json`.

Objectives: `throughput` (images/s, default), `efficiency` (images/s per GiB — the "best
ratio" number), `latency` (s/batch).

### Crop sizes follow the EAST detector grid

Widths come from the detector in
[`manuscript-ocr`](https://github.com/konstantinkozhin/manuscript-ocr/tree/main/src/manuscript/detectors/_east):
`EASTDataset(target_size=512, score_geo_scale=0.25)` normalises a page to a 512×512 canvas
and emits a 128×128 score map, i.e. a **4 px detection stride**, and `warp_quad` cuts crops
in canvas pixels. The recognizer reference config
(`recognizers/_trba/configs/config.json`) is `img_h=32, img_w=128, max_len=40`.

So `src/autotune.rs` uses:

* step = 4 px (EAST stride) × 8 (SEResNetQ width downsample, `T = img_w / 8`) = **32 px**:
  every candidate width is a multiple of 32, which keeps the crop boundary on the detector
  grid and every conv stage (maxpool 2×2, layer1 s2, layer3 s2, conv_out 2×2) integral;
* the ladder `128…512` in 32 px steps (inside one EAST canvas), then `640…1024` in 128 px
  steps (up to two canvases) so the expensive end is not oversampled;
* a floor of `max_len × 8`: with `T = img_w / 8` feature steps CTC cannot emit more than `T`
  symbols, so `max_len=40` needs `img_w ≥ 320` and `max_len=64` needs `img_w ≥ 512`.
  The reference `img_w=128` only fits `max_len ≤ 16` — worth knowing before copying that
  config into a run with long lines.

## Checkpoints and resuming

```bash
# каждые 200 батчей + в конце каждой эпохи (по умолчанию — каждые 500 батчей)
./target/release/trba-rust train.csv val.csv exp --ckpt-every-batch 200 --ckpt-every 2

# тот же запуск после Ctrl-C / OOM / падения драйвера продолжает с места обрыва
./target/release/trba-rust train.csv val.csv exp --ckpt-every-batch 200
#   📂 resume: exp/checkpoint_last.bin
#      завершено эпох: 3 | продолжать с эпохи 4, батч 201 | шаг оптимизатора 42201 |
#      Adam-моменты: восстановлены
```

`--ckpt-every-batch N` (default 500, `0` = off) writes the rolling `<exp_dir>/checkpoint_last.bin`
every N batches, and it is written again at the end of every epoch. Writes are atomic
(`.tmp` + `rename`), so interrupting a save cannot corrupt the previous one.
`--ckpt-every N` additionally keeps numbered files, with the epoch loss in the name, so the curve
is visible straight from `ls`:

```
exp/checkpoint_epoch003_loss4.1234.bin
exp/checkpoint_epoch004_loss3.9871.bin
```

### `best.bin`

Every validation (`--eval-every N`, off by default only if `val.csv` is missing) that improves the
metric writes a file whose name carries both numbers, and links `best.bin` to it:

```
exp/checkpoint_best_cer12.345_loss3.9871_epoch004.bin    ← the real file
exp/best.bin                                             ← hard link to it
```

`best.bin` is a **hard link**, not a copy: both names are one inode, so the 553 MB is stored once
(`ls -i exp` shows the same inode, `du` counts it once). The previous best file is deleted, so
improvements do not pile up. On filesystems without hard links (FAT/exFAT, some network shares) the
code falls back to a copy and says so in the log.

The selection metric is **CER**: the reference `validate()` never computed a validation loss (it
prints `acc`/`CER`, and its `loss` variable was dead code), while CER is what OCR quality is
measured by. The training loss of the epoch is still in the file name. On resume the best CER is
parsed back out of the existing file name, so `best.bin` is rewritten only on a real improvement
and not on the first validation after a restart.

`--autotune` probes never write checkpoints. A probe *is* one epoch, so the epoch-end save used to
add ~500 MB of D2H traffic plus a 553 MB disk write to every trial — which showed up as ~15 MB per
batch of "PCIe" in the autotune report. `TRBA_AUTOTUNE_PROBE=1` (set by the parent) disables the
rolling save, the numbered save and the banner.

Resume is automatic: the same command line picks the newest checkpoint in `exp_dir` (by mtime —
the rolling file or a numbered one, whichever is newer), `--resume PATH` picks a specific file,
`--no-resume` starts over. Training continues at `epoch_done + 1` and **skips the batches of that
epoch that were already done**, so an interrupted epoch is not repeated from the start.

Format `TRBA_V5` holds: `epoch_done`, `batch_in_epoch`, `Adam.t`, every parameter tensor
*with its Adam m/v*, BN running stats, and — in device mode — the conv Adam moments pulled back
from the GPU (they never live on the host otherwise). Size ≈ 553 MB for the 42M-parameter conv
stack, so keep a couple of GB free next to `exp_dir`; only one rolling file accumulates.

Two details make a mid-epoch resume exact rather than approximate:

* the batch order is a function of `(seed, epoch)`, not of the whole shuffle history, so the
  batches skipped after a resume are the same ones already trained on;
* Adam's `m`/`v` are restored on both sides (host parameters *and* the device conv state).
  Restoring only `t` restarts both moving averages from zero while the bias correction keeps
  dividing by `1 - beta^t ≈ 1`, so the first steps after a resume are effectively unnormalised
  and can throw the weights away — that is what the old `TRBA_V4` checkpoint did.

`TRBA_V4` files still load (weights, BN, `t`); the run says so explicitly instead of pretending
the moments were restored. If a checkpoint does not fit the current model (different `--hidden`,
different charset size), the loader reports the mismatch and the model is rebuilt from scratch —
a partially-applied file is never trained on.

## Recognising images with a trained checkpoint (`--infer`)

```bash
# каталог картинок → stdout; чекпоинт берётся из exp/best.bin
./target/release/trba-rust crops "" exp --charset charset.txt --img-w 512 --max-len 64 --infer

# CSV (первая колонка — пути, то есть годится val.csv) → TSV-файл
./target/release/trba-rust val.csv results.tsv exp --charset charset.txt --img-w 512 --infer

# конкретный чекпоинт и батч
./target/release/trba-rust crops out.tsv exp --charset charset.txt --infer \
    --resume exp/checkpoint_epoch012_loss3.4210.bin --batch-size 16
```

In `--infer` mode the positional arguments mean: `[0]` — one image, a directory of images, or a
CSV/TSV/TXT whose first column holds paths (so `train.csv`/`val.csv` work unchanged); `[1]` — the
results file, one `path<TAB>text` per line (empty string or nothing at all → stdout); `[2]` —
`exp_dir`, the same slot as in training. The checkpoint is `--resume PATH`, else
`<exp_dir>/best.bin`, else the newest `checkpoint_*.bin` in `exp_dir`. A failed load is fatal:
recognising with half a checkpoint is worse than an error.

Preprocessing and decoding are exactly the validation ones — aspect-preserving resize, white
canvas, `(p/255*2-1)`, int8 with scale 1/127, CNN → BiLSTM → greedy attention decode — so the text
matches what training reported. `--charset/--img-h/--img-w/--max-len/--hidden` must match the
checkpoint; a mismatch is reported instead of silently producing garbage. Unreadable files are
skipped with a warning and counted separately, and images are batched by `--batch-size`.

This is the piece the per-scribe loop needs: fine-tune on one scribe's crops and check the result
without leaving the binary (and later without Python at all).

## Layout convention: `[B, T, C]` outside the CNN

The CNN works in NCHW and hands the encoder `feat [B, T, 512]` (`T = img_w / 8`, mean over the
remaining height). The BiLSTM stack, the attention decoder and CTC all read and write `[B, T, ·]`,
and `cnn.backward` expects `d_feat` in the same layout. Inside `rnn.rs` the per-direction traces are
`[T, B, ·]` — the natural order for a time loop — but they never cross the module boundary.

The test `bilstm_batch_items_do_not_mix_forward_or_backward` pins this down: an item inside a batch
must produce exactly the output *and* the input gradient that it produces on its own (`b = 1`).

## Where the batch time goes (`TRBA_PROFILE=1`)

```bash
TRBA_PROFILE=1 ./target/release/trba-rust train.csv val.csv exp --batch-size 8 --img-w 512
```

At the end of every epoch a per-batch breakdown goes to stderr (format illustrated with made-up
numbers):

```
[profile] эпоха 1: 8 батч(ей), 1070.4 мс/батч | CPU 902.1 мс (84%), GPU 151.0 мс (14%), PCIe 17.3 мс (2%)
   img_h2d         2.31 мс/батч   0.2%  [pcie] загрузка int8-батча на GPU
   cnn_fwd        41.20 мс/батч   3.8%  [gpu]  CNN forward: int8 IMMA-ядра
   feat_d2h        7.10 мс/батч   0.7%  [pcie] выгрузка feat [B,T,512] + sync
   rnn_fwd       212.44 мс/батч  19.8%  [cpu]  BiLSTM forward (2 слоя x 2 направления)
   rnn_bwd       486.02 мс/батч  45.4%  [cpu]  BiLSTM backward
   cnn_bwd       102.55 мс/батч   9.6%  [gpu]  CNN backward (int8) + sync
   clip_adam      31.18 мс/батч   2.9%  [cpu]  clip по норме + Adam по host-параметрам
   adam_dev        5.02 мс/батч   0.5%  [gpu]  Adam по 42.15M conv-весам (GPU) + requant
   [int8] CNN forward : 14.196 GMAC/батч за 41.20 мс -> 345 GMAC/s (690 GOPS)
   [int8] CNN backward: 28.392 GMAC/батч за 102.55 мс -> 277 GMAC/s (554 GOPS)
```

Why this exists: only the CNN runs on the GPU. The BiLSTM, the attention decoder, CTC and Adam for
the non-conv parameters are scalar Rust on the host, and their cost grows with `batch × T`
(`T = img_w / 8`). That is why `nvidia-smi` can show 9–17 % utilisation with a nearly empty VRAM:
the card is not slow, it is waiting for the host. Doubling `img_w` doubles the host work, which is
exactly what the `sec/batch` column of `--autotune` shows.

How to read it:

* **CPU ≫ GPU** (the usual case here) → the fix is on the host side: parallelise the RNN/attention
  over the batch dimension (rayon), and only then raise the batch size. Until the host part is
  parallel, a bigger batch costs proportionally more host time, so images/second stays flat while
  VRAM stays empty.
* **GPU ≫ CPU** → compare the `[int8]` lines with the IMMA peak, `SM × 1024 MAC/clock × clock`
  (56 SM @ 1.4 GHz ≈ 40 TMAC/s ≈ 80 TOPS). A few hundred GMAC/s means the GEMM runs far below
  peak (small tiles, low occupancy) and is worth optimising; tens of TMAC/s means it is fine.
* **PCIe large** → see the PCIe section; a PCIe 2.0 x4 link gives ~2 GB/s.

The profiler calls `cudaStreamSynchronize` at every phase boundary — kernel launches are
asynchronous, so without it GPU time would be attributed to the next phase. The attribution is
honest, the profiled run itself is slightly slower, and nothing is measured unless
`TRBA_PROFILE=1`.

### Proving that INT8 uses the tensor cores

Three independent checks, cheapest first:

1. **At startup** the backend announces the path it picked:

   ```
   [trba] int8 GEMM: WMMA m16n16k16 s8*s8->s32 (тензорные ядра, IMMA) | cc 7.5 | SM 56 | выбран автоматически
   ```

   `launch_gemm_int8` takes the WMMA path when `cc ≥ 7.2` (Turing and newer) and the `__dp4a`
   kernel otherwise: DP4A here is a *fallback for pre-Turing cards*, not the training path.

2. **A/B on your own card** — `TRBA_INT8_GEMM=dp4a` forces the CUDA-core fallback,
   `TRBA_INT8_GEMM=wmma` forces the tensor-core path. Run both with `TRBA_PROFILE=1` and compare
   `cnn_fwd`/`cnn_bwd` and `s/batch`. If the numbers are the same, the tensor cores are not doing
   the work; the ratio is also a direct measure of how much INT8/IMMA buys you.

3. **In the compiled binary**:

   ```bash
   cuobjdump -sass target/release/build/trba-rust-*/out/libtrba_cuda.a | grep -c IMMA
   ```

   The PTX produced for `compute_75` contains 8 ×
   `wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32` (plus `wmma.load.*.s8` /
   `wmma.store.d.*.s32`) inside `gemm_int8_wmma_kernel`, and no `dp4a` there at all — all 23
   `dp4a` instructions live in `gemm_int8_dp4a_kernel`, which an `sm_75` device never dispatches.

`launch_gemm_fp16` is declared in `src/backend.rs` but never called: the fp16 GEMM is not part of
the training path (and its PTX is the f32-emulated WMMA form, i.e. CUDA cores, not tensor cores).

## BiLSTM on the GPU (`TRBA_GPU_RNN=1`)

By default only the CNN runs on the card; the BiLSTM is scalar Rust on the host and is the
single biggest term in the batch time (`rnn_fwd` + `rnn_bwd` in the profile, ~181 ms per batch at
`b=2, T=16`, ~314 ms at `b=1, T=64`). `TRBA_GPU_RNN=1` moves it to the device.

```bash
TRBA_GPU_RNN=1 TRBA_PROFILE=1 ./target/release/trba-rust train.csv val.csv exp \
    --img-w 512 --batch-size 8 --smoke 20
```

Stage 1 is a hybrid, and the split is deliberate:

* **on the GPU** — both directions of both LSTM layers: `lstm_fwd_step_kernel` (a warp per
  `(batch, output)`, four `__shfl_down_sync` reductions per step), `lstm_bwd_cell_kernel`,
  `lstm_bwd_rec_kernel`, and three weight-gradient kernels launched once per direction after the
  time loop (`lstm_wgrad_ih_kernel`, `lstm_wgrad_hh_kernel`, `lstm_bgrad_kernel`);
* **on the host** — the projection `cat -> y` and its backward. It is a single GEMM with no
  recurrence, so it is cheap, and both paths now share the same code (`cat_from_h`,
  `proj_forward`, `proj_backward_dcat`, `proj_grads`, `split_dcat`).

What that buys on the data path: `feat` from `cnn.forward` is handed straight to
`rnn.forward_device` and `d_feat` goes straight into `cnn.backward` — neither is ever downloaded
or uploaded. Only `enc` (for attention/CTC) and `dy` cross the link, ~0.13 MB at `b=2, T=16`
instead of ~200 MB in host mode.

What it costs: the LSTM weights go up and their gradients come down every batch —
10.5 MB + 10.5 MB for `hidden=256` with two layers and two directions, so **~21 MB per batch**
(measured with `TRBA_PCIE_COUNTER=1`: +20.5 MB per batch). On PCIe 2.0 x4 that is ~13 ms, against
the 181-314 ms of host BiLSTM it replaces, and unlike the host cost it does not grow with
`batch × T` — at `b=8` the saving is ~95 % of the RNN time. In the default configuration, where
the conv weights and their Adam state already live on the device (0.64 MB per batch), those
21 MB become the dominant PCIe consumer; stage 2 makes the LSTM weights, gradients and optimizer
state device-resident the same way and removes them.

Conventions the kernels rely on (identical to the host path, which is why the tests transfer):
`x`/`dx` are `[B, T, D]`; `gates` is `[T, B, 4H]` post-activation; `c`/`h`/`dh_out` are
`[T, B, H]` in **slot order**; `c_prev`/`h_prev` are read from slot-1 of the same trace (slot 0
means zeros), so there are no separate state buffers, no races and no atomics. Kernels take
single-direction pointers plus a `rev` flag and are launched per direction, which keeps
`slot -> tt = rev ? T-1-slot : slot` in exactly one place. Gate order is PyTorch's `(i, f, g, o)`.

The flag is off by default. Verification:

```bash
cargo test --release device_bilstm_matches_host_bilstm -- --test-threads=1   # enc, dx and every gradient
cargo test --release -- --test-threads=1                                     # 36/36
```

`device_bilstm_matches_host_bilstm` compares the device path against the host path for `enc`, `dx`
and all of `g_w_ih`, `g_w_hh`, `g_b_ih`, `g_b_hh` (both directions, both layers) plus the
projection gradients, at a relative tolerance of `1e-4` (fp32 summation order differs: warp
reduction vs a sequential loop). Without a GPU it exercises the CPU mirrors in
`cuda/cpu_backend.cpp`; on the card the same test exercises the real kernels. Training was also
compared end to end from an identical checkpoint: host path `loss 3.2462 (attn 3.1772 ctc 3.4218)`,
device path `loss 3.2463 (attn 3.1773 ctc 3.4219)` (the ~1e-4 gap is fp32 reassociation: the
weight-gradient kernels sum with four independent accumulators instead of one dependent chain).

SM occupancy: the forward kernel launches `B*H` warps (65 536 threads at `b=8, hidden=256`) and the
weight-gradient kernels launch `4H*D` / `4H*H` threads, so those fill all 56 SMs. The two per-slot
backward kernels are bounded by the recurrence itself — one step has exactly `B*H` independent
units of work — so they occupy `B*H/256` and `B*(D+H)/256` blocks: 8 and 24 at `b=8`, 32 and 128
at `b=32`. A bigger batch fills the card; filling it at a small batch needs a parallel scan over
time, which is a later stage.

Do not read speed from a CPU-backend run: there the device path is scalar mirror code and is
*slower* than the host path. It checks correctness, not performance.

## Muon instead of Adam

Matrices are optimised with **momentum-free Muon** (`src/conv2d.rs::muon_update` semantics, now
used everywhere), vectors — biases and BN `gamma`/`beta` — stay on Adam, for which Muon degenerates
into normalised SGD. Per matrix `[rows, cols]`:

```text
scale  = min( sqrt(min(rows,cols)) / ||g||_F , 10 )
lr_eff = lr * sqrt(max(rows,cols))          # autoscale, TRBA_MUON_AUTOSCALE=0 disables
w     -= lr_eff * scale * g ;  w *= 1 - lr*wd   # weight decay is decoupled
```

The autoscale is not cosmetic: since `n = rows*cols`, the per-element step magnitude works out to
`lr` — the same order as Adam. Without it the step would be 23-32x smaller (for 1024x512) and
training would crawl.

The conv weights (42.15M, device-resident) go through the same maths on the GPU:
`muon_step_f32_kernel` computes the scale from a device-side f64 norm scalar, so **no byte returns
to the host and no sync is needed**. Muon keeps no state, therefore `m`/`v` are not allocated at
all: **-338 MB of VRAM**, and checkpoints shrank from 553 324 344 to **184 591 952 bytes** (3x
smaller), which matters with `--ckpt-every-batch`.

The global `clip_grad_norm_(5.0)` runs only for the first `--clip-warmup` steps (default 500):
after that the Muon normalisation with its cap bounds the step, and skipping the clip also removes
one device sync per batch. `Adam::sum_sq` and `Adam::scale_grads` are parallel and stay
bit-compatible with the device-side sum (the existing pipeline test compares them at 1e-12).

Measured on `clip_adam` (sandbox, 2 cores, `b=2, img_w=128`, conv weights on the host — the worst
case; in the default configuration the conv weights are on the device and the host part is much
smaller):

| configuration | `clip_adam` ms/batch |
|---|---|
| Adam as it was (f64 per element, single thread, clip always) | **392.17** |
| Adam, but parallel (`TRBA_OPT=adam`) | 177.30 |
| **Muon fp32, clip during warmup** | **60.32** |
| **Muon fp32, clip off (steady state)** | **40.37** |
| MuonQ int8/128, clip off | 121.03 |

That is 392 -> 40 ms, 9.7x, on 2 cores; the parallel part scales with cores, so the absolute
numbers on your machine will be lower and the ratio similar.

### MuonQ is a regulariser here, not a speedup

`TRBA_MUON_Q=1` quantises the gradient to int8 with a scale per `TRBA_MUON_Q_BLOCK` (default 128)
elements, `max|g|/127`, half-away rounding. `||g||_F^2` is then computed as `sum(s_b^2 * sum(q^2))`
with the inner sum in **i64** — exactly, so the norm no longer depends on fp32 summation order.
Quality is checked by `muon_q_block128_tracks_fp32_muon`: cosine between the fp32 and int8 update
directions is 0.999994 on a 37x259 matrix with one block deliberately 40x larger than the rest
(per-tensor scaling breaks on that shape, per-block does not).

But on the host MuonQ is **slower** than fp32 Muon (140.85 vs 60.32 ms/batch), and the reason is
concrete: the gradient already sits in RAM as fp32, so quantising adds two passes (block max, int8
store) and the cheaper final read does not pay for them. The "less memory, more cache hits" argument
applies when the gradient is *stored or transported* in int8 rather than quantised in place — and
this project has exactly one such place, the important one: PCIe. With stage 2 of the GPU-RNN plan
the LSTM weight gradients travel device->host, 10.5 MB per batch; as int8 that is 2.6 MB, so MuonQ
turns from overhead into ~8 MB per batch of savings (~5 ms on PCIe 2.0 x4). The quantisation noise
as a regulariser stands regardless of transport.

So: `TRBA_MUON_Q=1` is off by default, switch it on deliberately and compare CER.

```bash
cargo test --release muon -- --test-threads=1     # 3 tests: formula, MuonQ, device kernel
TRBA_OPT=adam TRBA_PROFILE=1 ./target/release/trba-rust ...   # A/B against the old optimiser
```

If the loss rises instead of falling, the first things to change are `TRBA_MUON_CAP` (e.g. down to
1.0) and a longer clip (`--clip-warmup 5000`).

## Debugging a GPU fault

`TRBA_CUDA_DEBUG_SYNC=1` synchronises after every launch and memcpy and attributes the fault
to the launcher that caused it:

```bash
TRBA_CUDA_DEBUG_SYNC=1 ./target/release/trba-rust train.csv --smoke 2 --batch-size 4 --img-w 256
# [CUDA FATAL] cuda/kernels.cu:611: the kernel/memcpy launched just above faulted:
#              an illegal memory access was encountered
```

Without it the same fault surfaces wherever CUDA next happens to touch the driver — a
`cudaFree` or a `cudaMemcpyAsync` hundreds of launches later, which is how it ends up
looking like the memory manager is broken. It costs a few × in speed, so use it to locate a
crash, not for training. `compute-sanitizer --tool memcheck ./target/release/trba-rust …`
gives the same answer with the exact kernel and address.

Low-RAM boxes: every unit test builds the full 42M-parameter model, so run
`cargo test --release -- --test-threads=1` if the parallel run gets OOM-killed.

## Checking kernel logic without a GPU

`tools/kernel_logic_check.cpp` re-implements the tiling, the bounds guards and the barrier
schedule of the CUDA kernels on the CPU and compares them against a naive reference
(16 checks, ~2 s, needs only g++). It catches the class of bug the CPU backend cannot see:
uncovered output tiles, reads past the padded operands, divergent `__syncthreads()`.

```bash
make -f tools/Makefile logic    # CPU-side index arithmetic
make -f tools/Makefile cuda     # just compile cuda/kernels.cu (nvcc, no GPU needed)
```

## Toolkit notes

* glibc ≥ 2.41 (Debian 13, Ubuntu 25.10+) declares `sinpi`/`cospi` in a way that CUDA ≤ 12.9
  rejects in the host pass. `build.rs` detects the failure and retries `nvcc` with
  `-Xcompiler -U_GNU_SOURCE`; nothing in this project uses those functions.
* The static runtime is `libcudart_static.a` (there is no `libcudart.a`), and it needs
  `libculibos.a`. `TRBA_CUDA_SHARED_RT=1` links `libcudart.so` instead.
* `TRBA_CUDA_ARCH` accepts a list (`"75;86;89"`); one PTX image is embedded for the newest
  entry so unseen GPUs can be JITted. Without it `build.rs` asks `nvidia-smi`, and failing
  that builds a 75/86/89 fatbin.
