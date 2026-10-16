//! Optuna-подобный подбор пары (batch_size, img_w) под конкретную GPU.
//!
//! Цель — найти соотношение размера батча и длины картинки, при котором обучение
//! идёт быстрее всего и не упирается в видеопамять.  На PCIe 2.0 x4 трафик уже не
//! является ограничением (~0.6 MB на батч в device-режиме), поэтому оптимум
//! определяется VRAM и латентностью мелких передач: чем больше батч, тем меньше
//! передач на картинку.
//!
//! Две причины, по которым это не цикл внутри одного процесса:
//!
//! 1. `an illegal memory access was encountered` и CUDA OOM убивают процесс
//!    целиком, а контекст после них непригоден для дальнейших замеров.  Поэтому
//!    каждая проба — **отдельный дочерний процесс** с `--smoke K`: если
//!    конфигурация падает, родитель видит это по коду возврата и продолжает
//!    поиск, а не теряет весь прогон.
//! 2. Замер должен включать реальный путь данных (загрузка/квантование картинок,
//!    RNN+attention+CTC на хосте, PCIe), поэтому проба гоняет настоящие батчи, а
//!    не синтетические ядра.
//!
//! Поиск — случайный с бюджетом (аналог `RandomSampler` в Optuna): если бюджет
//! покрывает сетку, идёт полный перебор от мелких конфигураций к крупным (так
//! первые результаты появляются быстро и падение крупной пробы не обрывает
//! прогон), иначе — детерминированная по `--seed` выборка без возврата.
//!
//! ```text
//! trba-rust train.csv --autotune --autotune-trials 24 --autotune-batches 5
//! trba-rust train.csv --autotune --autotune-objective efficiency --autotune-vram 8000
//! ```

use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// ---- стандартные размеры EAST -------------------------------------------------
// Источник: manuscript-ocr, src/manuscript/detectors/_east/dataset.py и
// src/manuscript/recognizers/_trba/configs/config.json.

/// `EASTDataset(target_size=512)`: страница приводится к канве 512x512, кропы
/// (`warp_quad`) вырезаются в пикселях этой канвы.
pub const EAST_TARGET_SIZE: usize = 512;
/// `score_geo_scale=0.25`: карта детектора 128x128.
pub const EAST_GRID: usize = 128;
/// Шаг сетки детектора в пикселях канвы: 512 / 128 = 4.
pub const EAST_STRIDE: usize = EAST_TARGET_SIZE / EAST_GRID;
/// Референсные размеры распознавателя из config.json того же репозитория.
pub const REF_IMG_H: usize = 32;
pub const REF_IMG_W: usize = 128;
pub const REF_MAX_LEN: usize = 40;
/// SEResNetQ понижает ширину в 8 раз: T = img_w / 8 шагов признака.
pub const W_DOWNSAMPLE: usize = 8;

/// Шаг сетки ширины кропа = 4 px шага EAST * 8 понижения ширины = 32 px.
///
/// При таком шаге ширина кропа остаётся согласованной с сеткой детектора (граница
/// кропа попадает в ячейку 4 px, увеличенную в 8 раз свёрткой), и все стадии
/// SEResNetQ — maxpool 2x2, layer1 stride 2, layer3 stride 2, conv_out 2x2 —
/// дают целые размеры без потерь на округление.
pub fn width_step() -> usize {
    EAST_STRIDE * W_DOWNSAMPLE
}

/// Минимальная ширина, при которой CTC физически способен выдать `max_len`
/// символов: шагов признака T = img_w / 8, и нужно хотя бы `max_len` из них.
pub fn min_width(max_len: usize) -> usize {
    let need = max_len.saturating_mul(W_DOWNSAMPLE).max(1);
    round_up(need, width_step()).max(REF_IMG_W)
}

/// Максимальная ширина: две канвы EAST (512 * 2 = 1024) — дальше кроп строки
/// настолько сжат, что распознавание теряет смысл, а VRAM растёт квадратично.
pub fn max_width() -> usize {
    2 * EAST_TARGET_SIZE
}

/// Лестница ширин, согласованная с EAST: до 512 шаг 32 px (сетка детектора),
/// выше 512 шаг 128 px, чтобы не перебирать заведомо дорогие варианты.
/// Всё, что короче `min_width(max_len)`, отбрасывается — CTC там не сойдётся.
pub fn candidate_widths(max_len: usize) -> Vec<usize> {
    let step = width_step();
    let coarse = step * 4;
    let lo = min_width(max_len);
    let hi = max_width().max(lo);
    let mut v: Vec<usize> = (step..=EAST_TARGET_SIZE.min(hi))
        .step_by(step)
        .chain(((EAST_TARGET_SIZE + coarse)..=hi).step_by(coarse))
        .filter(|w| *w >= lo)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Лестница размеров батча.  Начинается с 1, потому что именно на малых батчах
/// видна латентность PCIe, а заканчивается 32 — дефолтом репозитория.
pub fn candidate_batches() -> Vec<usize> {
    vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
}

fn round_up(x: usize, m: usize) -> usize {
    if m == 0 {
        return x;
    }
    ((x + m - 1) / m) * m
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Objective {
    /// Картинки в секунду — максимум.
    Throughput,
    /// Картинки в секунду на GiB занятой памяти — максимум (лучшее соотношение).
    Efficiency,
    /// Секунд на батч — минимум (важно при фиксированном шаге логов/eval).
    Latency,
}

impl Objective {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "throughput" | "speed" | "img" => Ok(Objective::Throughput),
            "efficiency" | "ratio" | "vram" => Ok(Objective::Efficiency),
            "latency" | "time" => Ok(Objective::Latency),
            other => Err(anyhow!(
                "--autotune-objective {other}: ожидается throughput | efficiency | latency"
            )),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Objective::Throughput => "throughput (картинок/с)",
            Objective::Efficiency => "efficiency (картинок/с на GiB)",
            Objective::Latency => "latency (с/батч)",
        }
    }
    /// Больше — лучше; `None`, если конфигурация не прошла фильтр.
    fn score(&self, p: &Probe) -> Option<f64> {
        if !p.ok || p.img_per_s <= 0.0 {
            return None;
        }
        match self {
            Objective::Throughput => Some(p.img_per_s),
            Objective::Efficiency => {
                let gib = p.vram_mib / 1024.0;
                if gib > 1e-6 {
                    Some(p.img_per_s / gib)
                } else {
                    // CPU-бэкенд или карта без cudaMemGetInfo: VRAM неизвестна,
                    // сравниваем по пропускной способности.
                    Some(p.img_per_s)
                }
            }
            Objective::Latency => Some(-p.sec_per_batch),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Probe {
    pub batch: usize,
    pub img_w: usize,
    /// сколько батчей дочерний процесс реально успел прогнать
    pub batches: usize,
    pub ok: bool,
    pub note: String,
    pub sec_per_batch: f64,
    pub img_per_s: f64,
    pub vram_mib: f64,
    pub h2d_mb: f64,
    pub d2h_mb: f64,
}

impl Probe {
    /// Счётчик PCIe печатается за весь процесс, а сравнивать конфигурации нужно
    /// по стоимости одного батча — иначе проба с большим `--autotune-batches`
    /// выглядит дороже.
    fn per_batch(&self, total_mb: f64) -> f64 {
        if self.batches > 0 {
            total_mb / self.batches as f64
        } else {
            f64::NAN
        }
    }
    fn pcie_mb_per_batch(&self) -> f64 {
        let t = self.h2d_mb + self.d2h_mb;
        if t.is_finite() {
            self.per_batch(t)
        } else {
            f64::NAN
        }
    }
    fn failed(batch: usize, img_w: usize, note: impl Into<String>) -> Self {
        Probe {
            batch,
            img_w,
            batches: 0,
            ok: false,
            note: note.into(),
            sec_per_batch: f64::NAN,
            img_per_s: 0.0,
            vram_mib: f64::NAN,
            h2d_mb: f64::NAN,
            d2h_mb: f64::NAN,
        }
    }
}

pub struct Cfg {
    /// Сколько конфигураций попробовать (0 = вся сетка).
    pub trials: usize,
    /// Явный список ширин (`--autotune-widths 128,256,512`); пустой = лестница EAST.
    pub widths: Vec<usize>,
    /// Явный список размеров батча (`--autotune-batch-sizes 2,8,32`); пустой = лесенка.
    pub batch_sizes: Vec<usize>,
    /// Сколько батчей гоняет одна проба.
    pub probe_batches: usize,
    /// Таймаут одной пробы.
    pub timeout: Duration,
    /// Предел занятой памяти в MiB (0 = 90% от того, что видит cudaMemGetInfo).
    pub vram_limit_mib: f64,
    pub objective: Objective,
    /// Куда положить отчёт (JSON).
    pub out: String,
    pub seed: u64,
    pub max_len: usize,
    pub img_h: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            trials: 0,
            widths: Vec::new(),
            batch_sizes: Vec::new(),
            probe_batches: 4,
            timeout: Duration::from_secs(600),
            vram_limit_mib: 0.0,
            objective: Objective::Throughput,
            out: "autotune.json".into(),
            seed: 42,
            max_len: REF_MAX_LEN,
            img_h: REF_IMG_H,
        }
    }
}

/// `--autotune-widths 128,256` → [128, 256]; пустая строка → пустой список.
pub fn parse_list(s: &str) -> Result<Vec<usize>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split([',', ' ', ';'])
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim().parse::<usize>().map_err(|e| anyhow!("{t:?} не число: {e}"))
        })
        .collect()
}

/// Детерминированный ГПСЧ (xorshift64*), чтобы выборка воспроизводилась по --seed
/// и не тянуть зависимость только ради подбора.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // `seed | 1` смешивал соседние сиды в одно состояние (42|1 == 43|1),
        // поэтому разные --seed давали одинаковую выборку проб.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x632B_E59B_D51A_3E11) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Индекс в [0, n).
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

pub fn run(cfg: &Cfg) -> Result<()> {
    let exe = std::env::current_exe()?;
    let widths = if cfg.widths.is_empty() {
        candidate_widths(cfg.max_len)
    } else {
        let mut v = cfg.widths.clone();
        v.sort_unstable();
        v.dedup();
        v.retain(|w| *w > 0);
        v
    };
    let batches = if cfg.batch_sizes.is_empty() {
        candidate_batches()
    } else {
        let mut v = cfg.batch_sizes.clone();
        v.sort_unstable();
        v.dedup();
        v.retain(|b| *b > 0);
        v
    };
    if widths.is_empty() {
        return Err(anyhow!(
            "для max_len={} нет ни одной ширины, кратной {} и не короче {}",
            cfg.max_len,
            width_step(),
            min_width(cfg.max_len)
        ));
    }

    let (free0, total) = crate::backend::memory_info();
    let total_mib = total as f64 / 1048576.0;
    let limit = if cfg.vram_limit_mib > 0.0 {
        cfg.vram_limit_mib
    } else if total_mib > 0.0 {
        // 90% свободного на старте: запас на фрагментацию и на eval-проход,
        // который аллоцирует свои буферы поверх тренировочных.
        (free0 as f64 / 1048576.0) * 0.9
    } else {
        f64::INFINITY
    };

    println!(
        "🔎 --autotune: бэкенд {} | GPU {} | свободно {:.0} MiB из {:.0} MiB | предел {:.0} MiB",
        crate::backend::backend_name(),
        crate::backend::device_name(),
        free0 as f64 / 1048576.0,
        total_mib,
        limit
    );
    println!(
        "   сетка EAST: канва {EAST_TARGET_SIZE}px, шаг детектора {EAST_STRIDE}px, \
         шаг ширины кропа {}px (T = img_w/{W_DOWNSAMPLE}); референс распознавателя \
         {REF_IMG_H}x{REF_IMG_W}, max_len {REF_MAX_LEN}",
        width_step()
    );
    println!(
        "   ширины ({}): {:?} | батчи ({}): {:?}",
        widths.len(),
        widths,
        batches.len(),
        batches
    );
    println!(
        "   цель: {} | проб на конфигурацию: {} батчей | всего сетка {} | бюджет {}",
        cfg.objective.name(),
        cfg.probe_batches,
        widths.len() * batches.len(),
        if cfg.trials == 0 {
            "вся сетка".to_string()
        } else {
            format!("{} проб", cfg.trials)
        }
    );
    if cfg.max_len * W_DOWNSAMPLE > *widths.last().unwrap_or(&0) {
        println!(
            "   ⚠️  max_len={} требует img_w ≥ {}, а сетка ограничена {} — CTC не выдаст \
             полную длину на узких вариантах",
            cfg.max_len,
            min_width(cfg.max_len),
            max_width()
        );
    }

    // ---- порядок проб ----
    let mut grid: Vec<(usize, usize)> = Vec::new();
    // мелкие конфигурации первыми: они дёшевы и почти всегда проходят, поэтому
    // таблица наполняется сразу, а падение крупной пробы ничего не ломает
    for &b in batches.iter() {
        for &w in widths.iter() {
            grid.push((b, w));
        }
    }
    if cfg.trials > 0 && cfg.trials < grid.len() {
        // случайная выборка без возврата (Optuna RandomSampler)
        let mut rng = Rng::new(cfg.seed);
        for i in (1..grid.len()).rev() {
            grid.swap(i, rng.below(i + 1));
        }
        grid.truncate(cfg.trials);
    }

    let mut probes: Vec<Probe> = Vec::with_capacity(grid.len());
    let t_all = Instant::now();
    for (k, &(b, w)) in grid.iter().enumerate() {
        print!(
            "[{}/{}] batch={:<3} img_w={:<5} ... ",
            k + 1,
            grid.len(),
            b,
            w
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let p = probe(&exe, cfg, b, w);
        if p.ok {
            println!(
                "{} с/батч | {} карт/с | {} MiB | PCIe {} MB/батч{}",
                fmt(p.sec_per_batch, 2),
                fmt(p.img_per_s, 1),
                fmt(p.vram_mib, 0),
                fmt(p.pcie_mb_per_batch(), 2),
                if p.vram_mib > limit {
                    "  ⚠️ выше предела"
                } else {
                    ""
                }
            );
        } else {
            println!("❌ {}", p.note);
        }
        probes.push(p);
    }

    // ---- отбор ----
    let mut good: Vec<&Probe> = probes
        .iter()
        .filter(|p| p.ok && (p.vram_mib <= limit || !(p.vram_mib > 0.0)))
        .collect();
    good.sort_by(|a, b| {
        let (sa, sb) = (cfg.objective.score(a), cfg.objective.score(b));
        sb.unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&sa.unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n===== итог ({} пробе, {:.0} с) =====", probes.len(), t_all.elapsed().as_secs_f64());
    println!(
        "{:>5} {:>6} {:>10} {:>10} {:>9} {:>11} {:>9} {:>9}  {}",
        "батч", "img_w", "с/батч", "карт/с", "MiB", "H2D MB/б", "D2H MB/б", "PCIe MB/б", "статус"
    );
    for p in probes.iter() {
        if p.ok {
            println!(
                "{:>5} {:>6} {:>10} {:>10} {:>9} {:>11} {:>9} {:>9}  {}",
                p.batch,
                p.img_w,
                fmt(p.sec_per_batch, 3),
                fmt(p.img_per_s, 1),
                fmt(p.vram_mib, 0),
                fmt(p.per_batch(p.h2d_mb), 2),
                fmt(p.per_batch(p.d2h_mb), 2),
                fmt(p.pcie_mb_per_batch(), 2),
                if p.vram_mib > limit { "выше предела" } else { "ok" }
            );
        } else {
            println!(
                "{:>5} {:>6} {:>10} {:>10} {:>9} {:>11} {:>9} {:>9}  ❌ {}",
                p.batch, p.img_w, "-", "-", "-", "-", "-", "-", p.note
            );
        }
    }

    let best = good.first().copied();
    if let Some(b) = best {
        println!(
            "\n🏆 лучшее соотношение: --batch-size {} --img-w {} ({} карт/с, {} MiB, {} с/батч)",
            b.batch,
            b.img_w,
            fmt(b.img_per_s, 1),
            fmt(b.vram_mib, 0),
            fmt(b.sec_per_batch, 2)
        );
        println!(
            "   запуск: trba-rust <train.csv> [val.csv] [exp_dir] --batch-size {} --img-h {} --img-w {} --max-len {}",
            b.batch, cfg.img_h, b.img_w, cfg.max_len
        );
        if cfg.objective != Objective::Throughput {
            if let Some(fastest) = probes.iter().filter(|p| p.ok).max_by(|a, c| {
                a.img_per_s
                    .partial_cmp(&c.img_per_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                println!(
                    "   (максимум картинок/с: --batch-size {} --img-w {} → {} карт/с)",
                    fastest.batch,
                    fastest.img_w,
                    fmt(fastest.img_per_s, 1)
                );
            }
        }
    } else {
        println!(
            "\n⚠️  ни одна конфигурация не прошла.  Причины по пробам выше; если падения \
             выглядят как 'illegal memory access', запустите одну пробу с \
             TRBA_CUDA_DEBUG_SYNC=1 — она назовёт конкретное ядро."
        );
    }

    write_report(cfg, &probes, best.map(|b| (b.batch, b.img_w)), limit)?;
    println!("   отчёт: {}", cfg.out);
    Ok(())
}

/// Одна проба = дочерний процесс с `--smoke K`.
///
/// Аргументы родителя пробрасываются как есть (пути к train.csv/charset нужны
/// дочернему процессу для той же модели), но `--autotune*` убираются, а
/// `--smoke/--batch-size/--img-w` подставляются свои.
fn probe(exe: &std::path::Path, cfg: &Cfg, batch: usize, img_w: usize) -> Probe {
    let argv: Vec<String> = std::env::args().collect();
    let mut args: Vec<String> = Vec::new();
    let mut skip_value = false;
    for a in argv.iter().skip(1) {
        if skip_value {
            skip_value = false;
            continue;
        }
        if a == "--autotune" {
            continue;
        }
        if a.starts_with("--autotune-") {
            skip_value = true;
            continue;
        }
        if matches!(a.as_str(), "--smoke" | "--batch-size" | "-b" | "--img-w" | "--img-h") {
            skip_value = true;
            continue;
        }
        // чекпоинт в пробе не нужен: он только замедляет старт и может не подойти
        // по форме, если родитель запущен с другим img_w
        if matches!(a.as_str(), "--resume" | "-r" | "--no-resume") {
            continue;
        }
        args.push(a.clone());
    }
    args.extend([
        "--smoke".to_string(),
        cfg.probe_batches.to_string(),
        "--batch-size".to_string(),
        batch.to_string(),
        "--img-w".to_string(),
        img_w.to_string(),
        "--img-h".to_string(),
        cfg.img_h.to_string(),
    ]);

    // stderr — в отдельный файл: читать два pipe последовательно из одного
    // потока значит рисковать взаимной блокировкой, если дочерний процесс
    // заполнит буфер stderr, пока родитель занят stdout.
    let err_path = std::env::temp_dir().join(format!(
        "trba-autotune-err-{}-{}-{}.log",
        std::process::id(),
        batch,
        img_w
    ));
    let err_file = match std::fs::File::create(&err_path) {
        Ok(f) => f,
        Err(e) => {
            return Probe::failed(
                batch,
                img_w,
                format!("не создать {}: {e}", err_path.display()),
            )
        }
    };

    let mut child = match Command::new(exe)
        .args(&args)
        .env("TRBA_AUTOTUNE_PROBE", "1")
        .env("TRBA_PCIE_COUNTER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::from(err_file))
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&err_path);
            return Probe::failed(batch, img_w, format!("не удалось запустить пробу: {e}"));
        }
    };

    // читаем stdout дочернего процесса, пока он не закрыл pipe
    let mut out_lines: Vec<String> = Vec::new();
    if let Some(o) = child.stdout.take() {
        for line in BufReader::new(o).lines() {
            if let Ok(l) = line {
                out_lines.push(l);
            }
        }
    }

    let status = match child.wait_timeout(cfg.timeout) {
        Ok(s) => s,
        Err(note) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&err_path);
            return Probe::failed(batch, img_w, note);
        }
    };
    let err_text = std::fs::read_to_string(&err_path).unwrap_or_default();
    let _ = std::fs::remove_file(&err_path);

    // строка PROBE печатается дочерним процессом в конце smoke-эпохи
    let mut p = match out_lines.iter().find(|l| l.starts_with("PROBE ")) {
        Some(line) => parse_probe(line, batch, img_w),
        None => Probe::failed(
            batch,
            img_w,
            format!(
                "нет строки PROBE ({})",
                exit_reason(&status).unwrap_or_else(|| "процесс завершился".into())
            ),
        ),
    };

    if !p.ok {
        // причина: CUDA FATAL (TRBA_CUDA_DEBUG_SYNC), OOM, assert, сигнал ядра
        let reason = err_text
            .lines()
            .chain(out_lines.iter().map(|s| s.as_str()))
            .find(|l| {
                l.contains("CUDA FATAL")
                    || l.contains("illegal memory access")
                    || l.contains("out of memory")
                    || l.contains("panicked")
                    || l.contains("assertion")
                    || l.contains("CUDA error")
            })
            .map(|l| l.trim().chars().take(180).collect::<String>())
            // anyhow печатает причину как «Error: …» — это и есть ответ, почему
            // проба не дошла до строки PROBE (например, не найден charset.txt)
            .or_else(|| {
                err_text
                    .lines()
                    .find(|l| l.starts_with("Error:") || l.starts_with("error:"))
                    .map(|l| l.trim().chars().take(180).collect::<String>())
            })
            .or_else(|| exit_reason(&status));
        if let Some(r) = reason {
            p.note = format!("{} | {}", p.note, r);
        }
    }

    // трафик PCIe печатается отдельной строкой при TRBA_PCIE_COUNTER=1;
    // репортер зарегистрирован через atexit и пишет в stderr
    let pcie_line = err_text
        .lines()
        .chain(out_lines.iter().map(|l| l.as_str()))
        .find(|l| l.contains("[pcie]"));
    if let Some(l) = pcie_line {
        if let Some(v) = field_after(l, "H2D") {
            p.h2d_mb = v;
        }
        if let Some(v) = field_after(l, "D2H") {
            p.d2h_mb = v;
        }
    }
    p
}

/// Число для таблицы: прочерк вместо NaN/inf, иначе колонки «съезжают», а в
/// отчёте мелькают значения, которых никогда не существовало.
fn fmt(v: f64, digits: usize) -> String {
    if v.is_finite() {
        format!("{v:.digits$}")
    } else {
        "-".to_string()
    }
}

/// Чем именно умер дочерний процесс.  SIGKILL на Linux почти всегда означает,
/// что конфигурацию убил OOM-killer (не хватило VRAM или RAM) — это полезнее,
/// чем «код сигнал».
fn exit_reason(st: &std::process::ExitStatus) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            let what = match sig {
                9 => "SIGKILL — скорее всего OOM: конфигурации не хватило памяти",
                6 => "SIGABRT — CHECK_CUDA/CHECK_LAUNCH оборвал процесс (см. stderr)",
                11 => "SIGSEGV",
                _ => "сигнал",
            };
            return Some(format!("{what} ({sig})"));
        }
    }
    st.code().map(|c| format!("код возврата {c}"))
}

fn parse_probe(line: &str, batch: usize, img_w: usize) -> Probe {
    let get = |k: &str| -> f64 {
        line.split_whitespace()
            .find_map(|tok| {
                let mut it = tok.splitn(2, '=');
                let (key, val) = (it.next()?, it.next()?);
                if key == k {
                    val.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    Probe {
        batch,
        img_w,
        batches: get("batches").max(0.0) as usize,
        ok: get("ok") == 1.0,
        note: String::new(),
        sec_per_batch: get("sec_per_batch"),
        img_per_s: get("img_per_s"),
        vram_mib: get("vram_mib"),
        h2d_mb: f64::NAN,
        d2h_mb: f64::NAN,
    }
}

/// `[pcie] H2D 202.32 MB (11928 calls) | D2H 37.44 MB ...` → число после ключа.
fn field_after(line: &str, key: &str) -> Option<f64> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key || tok.trim_end_matches(':') == key {
            if let Some(num) = it.next() {
                return num.parse::<f64>().ok();
            }
        }
    }
    None
}

fn write_report(
    cfg: &Cfg,
    probes: &[Probe],
    best: Option<(usize, usize)>,
    limit_mib: f64,
) -> Result<()> {
    // JSON пишем руками: serde в проекте нет, а тянуть его ради отчёта не стоит
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"objective\": {:?},\n", cfg.objective));
    s.push_str(&format!("  \"img_h\": {},\n", cfg.img_h));
    s.push_str(&format!("  \"max_len\": {},\n", cfg.max_len));
    s.push_str(&format!("  \"width_step\": {},\n", width_step()));
    s.push_str(&format!("  \"east_target_size\": {},\n", EAST_TARGET_SIZE));
    s.push_str(&format!("  \"east_stride\": {},\n", EAST_STRIDE));
    s.push_str(&format!("  \"vram_limit_mib\": {:.1},\n", limit_mib));
    s.push_str(&format!("  \"probe_batches\": {},\n", cfg.probe_batches));
    s.push_str(&format!("  \"seed\": {},\n", cfg.seed));
    if let Some((b, w)) = best {
        s.push_str(&format!("  \"best\": {{\"batch_size\": {b}, \"img_w\": {w}}},\n"));
    } else {
        s.push_str("  \"best\": null,\n");
    }
    s.push_str("  \"probes\": [\n");
    for (i, p) in probes.iter().enumerate() {
        let num = |v: f64| {
            if v.is_finite() {
                format!("{v:.6}")
            } else {
                "null".to_string()
            }
        };
        s.push_str(&format!(
            "    {{\"batch_size\": {}, \"img_w\": {}, \"ok\": {}, \"sec_per_batch\": {}, \
             \"img_per_s\": {}, \"vram_mib\": {}, \"h2d_mb\": {}, \"d2h_mb\": {}, \"note\": {:?}}}{}",
            p.batch,
            p.img_w,
            p.ok,
            num(p.sec_per_batch),
            num(p.img_per_s),
            num(p.vram_mib),
            num(p.h2d_mb),
            num(p.d2h_mb),
            p.note,
            if i + 1 == probes.len() { "" } else { "," }
        ));
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    if let Some(dir) = std::path::Path::new(&cfg.out).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    std::fs::write(&cfg.out, s)?;
    Ok(())
}

/// `wait` с таймаутом без внешних зависимостей: опрашиваем статус, пока не
/// вышли за бюджет.  Проба, которая зависла (например, на бесконечном
/// `cudaDeviceSynchronize` после падения ядра), не должна останавливать поиск.
trait WaitTimeout {
    fn wait_timeout(&mut self, t: Duration) -> Result<std::process::ExitStatus, String>;
}

impl WaitTimeout for std::process::Child {
    fn wait_timeout(&mut self, t: Duration) -> Result<std::process::ExitStatus, String> {
        let start = Instant::now();
        loop {
            match self.try_wait() {
                Ok(Some(st)) => return Ok(st),
                Ok(None) => {
                    if start.elapsed() > t {
                        return Err(format!("таймаут {} с", t.as_secs()));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(format!("wait: {e}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_ladder_follows_the_east_grid() {
        // шаг сетки = 4 px детектора * 8 понижения ширины
        assert_eq!(width_step(), 32);
        assert_eq!(EAST_STRIDE, 4);
        let ws = candidate_widths(32);
        assert!(!ws.is_empty());
        assert!(ws.iter().all(|w| w % width_step() == 0), "все ширины кратны 32");
        assert!(ws.contains(&256), "рабочая ширина пользователя должна быть в сетке");
        // 128 (референс config.json) попадает в сетку только при max_len <= 16:
        // при T = img_w/8 = 16 шагов CTC длиннее не сойдётся
        assert!(candidate_widths(16).contains(&REF_IMG_W), "референсные 128 при max_len=16");
        assert!(!ws.contains(&REF_IMG_W), "а при max_len=32 их уже быть не должно");
        assert!(*ws.last().unwrap() <= max_width());
        // выше канвы EAST шаг укрупняется, чтобы не перебирать дорогие варианты
        let above: Vec<usize> = ws.iter().copied().filter(|w| *w > EAST_TARGET_SIZE).collect();
        for pair in above.windows(2) {
            assert_eq!(pair[1] - pair[0], width_step() * 4);
        }
    }

    #[test]
    fn min_width_respects_the_ctc_step_count() {
        // T = img_w / 8 шагов признака: для max_len=64 нужно >= 512
        assert_eq!(min_width(64), 512);
        assert_eq!(min_width(32), 256);
        assert_eq!(min_width(1), REF_IMG_W, "не короче референсных 128");
        assert!(candidate_widths(64).iter().all(|w| *w >= 512));
        // с max_len=40 (референс EAST-конфига) 128 уже не подходит: T=16 < 40
        assert!(!candidate_widths(REF_MAX_LEN).contains(&REF_IMG_W));
        assert_eq!(min_width(REF_MAX_LEN), 320);
    }

    #[test]
    fn probe_line_parses() {
        let l = "PROBE b=4 w=256 h=32 ok=1 batches=4 sec_per_batch=0.812 \
                 img_per_s=19.70 vram_mib=3120.0 vram_total_mib=10240.0";
        let p = parse_probe(l, 4, 256);
        assert!(p.ok);
        assert!((p.sec_per_batch - 0.812).abs() < 1e-9);
        assert!((p.img_per_s - 19.70).abs() < 1e-9);
        assert!((p.vram_mib - 3120.0).abs() < 1e-9);
    }

    #[test]
    fn pcie_line_parses() {
        let l = "[pcie] H2D 202.32 MB (11928 calls) | D2H 37.44 MB (22628 calls) | total 239.76 MB";
        assert_eq!(field_after(l, "H2D"), Some(202.32));
        assert_eq!(field_after(l, "D2H"), Some(37.44));
    }

    #[test]
    fn pcie_totals_are_normalised_per_batch() {
        let l = "PROBE b=4 w=256 h=32 ok=1 batches=8 sec_per_batch=0.5 img_per_s=8.0 \
                 vram_mib=3000.0 vram_total_mib=10240.0";
        let mut p = parse_probe(l, 4, 256);
        assert_eq!(p.batches, 8);
        p.h2d_mb = 8.0;
        p.d2h_mb = 4.0;
        assert!((p.per_batch(p.h2d_mb) - 1.0).abs() < 1e-12);
        assert!((p.pcie_mb_per_batch() - 1.5).abs() < 1e-12);
        // без счётчика батчей делить не на что — прочерк, а не деление на ноль
        p.batches = 0;
        assert!(p.pcie_mb_per_batch().is_nan());
    }

    #[test]
    fn objective_ranks_and_filters() {
        let mut ok = Probe::failed(4, 256, "");
        ok.ok = true;
        ok.sec_per_batch = 1.0;
        ok.img_per_s = 16.0;
        ok.vram_mib = 2048.0;
        let bad = Probe::failed(8, 512, "illegal memory access");
        assert_eq!(Objective::Throughput.score(&bad), None);
        assert_eq!(Objective::Throughput.score(&ok), Some(16.0));
        assert!((Objective::Efficiency.score(&ok).unwrap() - 16.0 / 2.0).abs() < 1e-9);
        assert_eq!(Objective::Latency.score(&ok), Some(-1.0));
        // objective.parse принимает три имени и отвергает остальное
        assert_eq!(Objective::parse("efficiency").unwrap(), Objective::Efficiency);
        assert!(Objective::parse("быстро").is_err());
    }

    #[test]
    fn table_prints_dash_instead_of_nan() {
        assert_eq!(fmt(f64::NAN, 2), "-");
        assert_eq!(fmt(f64::INFINITY, 2), "-");
        assert_eq!(fmt(1.5, 2), "1.50");
        assert_eq!(fmt(0.30680, 1), "0.3");
        assert_eq!(fmt(601.0, 0), "601");
    }

    #[test]
    fn list_parsing() {
        assert_eq!(parse_list("").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_list("128, 256;512 1024").unwrap(), vec![128, 256, 512, 1024]);
        assert!(parse_list("128,x").is_err());
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let seq_a: Vec<usize> = (0..8).map(|_| a.below(100)).collect();
        let seq_b: Vec<usize> = (0..8).map(|_| b.below(100)).collect();
        assert_eq!(seq_a, seq_b);
        let mut c = Rng::new(43);
        let seq_c: Vec<usize> = (0..8).map(|_| c.below(100)).collect();
        assert_ne!(seq_a, seq_c);
    }
}
