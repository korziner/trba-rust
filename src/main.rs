//! TRBA trainer (Rust + INT8 Tensor Cores), reference-faithful port of
//! `manuscript-ocr/recognizers/_trba/training/train.py`:
//!
//! * SEResNet31 CNN (INT8 fake-QAT GEMMs, fp32 BatchNorm/SE/residuals)
//! * 2-layer Bidirectional LSTM encoder
//! * attention decoder (teacher forcing) + CTC head
//! * loss = (1-w) * CE(attention, ignore <PAD>) + w * CTC, w decays 0.3 -> 0.03
//! * Adam + cosine LR + global grad-norm clip 5.0
//!
//! CSV format: `path,text` per line (paths relative to the CSV directory or
//! absolute); `charset.txt` next to the train CSV defines the vocabulary
//! (first three tokens must be <PAD> <SOS> <EOS>).

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use trba_rust::attn::AttentionDecoder;
use trba_rust::backend::*;
use trba_rust::ctc::ctc_loss_backward;
use trba_rust::opt::{Adam, ParamRef};
use trba_rust::qnet::{LinearF, Scratch, SEResNetQ};
use trba_rust::rnn::BiLstmStack;

const CKPT_MAGIC: &[u8; 8] = b"TRBA_V4\n";

// ------------------------------------------------------------------ args --

struct Args {
    train_csv: String,
    val_csv: Option<String>,
    exp_dir: String,
    charset: Option<String>,
    batch: usize,
    epochs: usize,
    lr: f32,
    weight_decay: f32,
    img_h: usize,
    img_w: usize,
    max_len: usize,
    hidden: usize,
    resume: Option<String>,
    no_resume: bool,
    ckpt_every: usize,
    seed: u64,
    smoke: usize,
    eval_every: usize,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        train_csv: String::new(),
        val_csv: None,
        exp_dir: "exp".into(),
        charset: None,
        batch: 32,
        epochs: 100,
        lr: 5.107e-4,
        weight_decay: 1.9548e-5,
        img_h: 32,
        img_w: 1024,
        max_len: 64,
        hidden: 256,
        resume: None,
        no_resume: false,
        ckpt_every: 0,
        seed: 42,
        smoke: 0,
        eval_every: 1,
    };
    let mut positional = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].clone();
        let take = |i: &mut usize, name: &str| -> Result<String> {
            if *i + 1 >= argv.len() {
                return Err(anyhow!("{name} требует значение"));
            }
            *i += 1;
            Ok(argv[*i].clone())
        };
        match arg.as_str() {
            "--help" | "-h" => {
                println!("trba-rust <train.csv> [val.csv] [exp_dir] [опции]");
                println!("  --batch-size N   --epochs N     --lr X        --weight-decay X");
                println!("  --img-h N        --img-w N      --max-len N   --hidden N");
                println!("  --charset PATH   --resume [PATH] --no-resume  --ckpt-every N");
                println!("  --seed N         --smoke N      --eval-every N");
                std::process::exit(0);
            }
            "--batch-size" | "-b" => a.batch = take(&mut i, &arg)?.parse()?,
            "--epochs" => a.epochs = take(&mut i, &arg)?.parse()?,
            "--lr" => a.lr = take(&mut i, &arg)?.parse()?,
            "--weight-decay" => a.weight_decay = take(&mut i, &arg)?.parse()?,
            "--img-h" => a.img_h = take(&mut i, &arg)?.parse()?,
            "--img-w" => a.img_w = take(&mut i, &arg)?.parse()?,
            "--max-len" => a.max_len = take(&mut i, &arg)?.parse()?,
            "--hidden" => a.hidden = take(&mut i, &arg)?.parse()?,
            "--charset" => a.charset = Some(take(&mut i, &arg)?),
            "--resume" | "-r" => match argv.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    i += 1;
                    a.resume = Some(v.clone());
                }
                _ => a.resume = Some(String::new()),
            },
            "--no-resume" => a.no_resume = true,
            "--ckpt-every" => a.ckpt_every = take(&mut i, &arg)?.parse()?,
            "--seed" => a.seed = take(&mut i, &arg)?.parse()?,
            "--smoke" => a.smoke = take(&mut i, &arg)?.parse()?,
            "--eval-every" => a.eval_every = take(&mut i, &arg)?.parse()?,
            other => {
                if other.starts_with('-') {
                    return Err(anyhow!("неизвестная опция: {other}"));
                }
                positional.push(other.to_string());
            }
        }
        i += 1;
    }
    if a.smoke == 0 {
        a.train_csv = positional.first().cloned().ok_or_else(|| anyhow!("не указан train.csv"))?;
        a.val_csv = positional.get(1).cloned();
        if let Some(d) = positional.get(2) {
            a.exp_dir = d.clone();
        }
    }
    Ok(a)
}

// ------------------------------------------------------------------ data --

struct Sample {
    path: String,
    tokens: Vec<usize>,
}

fn load_charset(path: &str) -> Result<Vec<String>> {
    let f = std::fs::File::open(path).map_err(|e| anyhow!("charset {path}: {e}"))?;
    let mut itos = Vec::new();
    for line in BufReader::new(f).lines() {
        let t = line?;
        if !t.is_empty() {
            itos.push(t);
        }
    }
    if itos.len() < 4 {
        return Err(anyhow!("charset слишком короткий"));
    }
    Ok(itos)
}

/// `path,text` with optional quoting ("" escape).
fn parse_csv_line(line: &str) -> Option<(String, String)> {
    let (p, rest) = if line.starts_with('"') {
        let end = line[1..].find('"').map(|i| i + 1)?;
        let mut p = line[1..end].replace("\"\"", "\"");
        let mut rest = &line[end + 1..];
        if rest.starts_with(',') {
            rest = &rest[1..];
        }
        p.shrink_to_fit();
        (p, rest)
    } else {
        let mut it = line.splitn(2, ',');
        (it.next()?.to_string(), it.next().unwrap_or(""))
    };
    let text = rest.trim().trim_matches('"').to_string();
    if p.is_empty() || text.is_empty() {
        return None;
    }
    Some((p, text))
}

fn load_csv(path: &str, stoi: &std::collections::HashMap<String, usize>, max_len: usize) -> (Vec<Sample>, usize) {
    let base = Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return (Vec::new(), 0),
    };
    let mut out = Vec::new();
    let mut skipped = 0;
    for line in BufReader::new(f).lines().flatten() {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let (rel, text) = match parse_csv_line(&line) {
            Some(v) => v,
            None => {
                skipped += 1;
                continue;
            }
        };
        // like pack_attention_targets: unknown chars dropped, sequence
        // truncated to max_len (NOT skipped)
        let ids: Vec<usize> = text
            .chars()
            .filter_map(|c| stoi.get(&c.to_string()).copied())
            .take(max_len)
            .collect();
        if ids.is_empty() {
            skipped += 1;
            continue;
        }
        let p = Path::new(&rel);
        let full = if p.is_absolute() { rel.clone() } else { base.join(rel).to_string_lossy().into_owned() };
        out.push(Sample { path: full, tokens: ids });
    }
    (out, skipped)
}

/// Reference preprocessing: RGB -> aspect-preserving resize -> white canvas
/// (align left / centre vertically) -> (p/255*2-1) -> int8 with scale 1/127.
fn load_image_q(path: &str, img_h: usize, img_w: usize) -> Option<Vec<i8>> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w0, h0) = img.dimensions();
    let scale = (img_h as f32 / h0.max(1) as f32).min(img_w as f32 / w0.max(1) as f32);
    let nw = (w0 as f32 * scale).round().max(1.0) as u32;
    let nh = (h0 as f32 * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    let mut buf = vec![255u8; img_h * img_w * 3];
    let y0 = (img_h.saturating_sub(nh as usize)) / 2;
    for y in 0..nh as usize {
        for x in 0..nw as usize {
            let px = resized.get_pixel(x as u32, y as u32);
            let o = ((y + y0) * img_w + x) * 3;
            buf[o] = px[0];
            buf[o + 1] = px[1];
            buf[o + 2] = px[2];
        }
    }
    Some(buf.iter().map(|&p| (((p as f32 / 255.0) * 2.0 - 1.0) * 127.0).round().clamp(-127.0, 127.0) as i8).collect())
}

// ---------------------------------------------------------------- metrics --

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + c);
        }
    }
    d[n][m]
}

// ----------------------------------------------------------------- model --

struct Model {
    cnn: SEResNetQ,
    rnn: BiLstmStack,
    attn: AttentionDecoder,
    ctc: LinearF,
}

impl Model {
    fn new(in_dim: usize, hidden: usize, classes: usize, sos: usize, eos: usize, pad: usize) -> Self {
        Self {
            cnn: SEResNetQ::new(),
            rnn: BiLstmStack::new(in_dim, hidden, 2),
            attn: AttentionDecoder::new(hidden, hidden, classes, sos, eos, pad),
            ctc: LinearF::new(hidden, classes, true),
        }
    }

    fn zero_grad(&mut self) {
        self.cnn.zero_grad();
        self.rnn.zero_grad();
        self.attn.zero_grad();
        self.ctc.zero_grad();
    }

    fn params(&mut self) -> Vec<ParamRef> {
        let mut v: Vec<ParamRef> = Vec::new();
        macro_rules! push {
            ($name:expr, $p:expr, $g:expr) => {
                let (p, g) = ($p, $g);
                let len = p.len();
                v.push(ParamRef { name: $name, p: p.as_mut_ptr(), g: g.as_ptr() as *mut f32, len });
            };
        }
        for c in self.cnn.convs_mut() {
            let n = c.name.clone();
            push!(format!("{n}.w"), &mut c.w, &mut c.g_w);
            push!(format!("{n}.gamma"), &mut c.bn.gamma, &mut c.bn.g_gamma);
            push!(format!("{n}.beta"), &mut c.bn.beta, &mut c.bn.g_beta);
        }
        for (li, layer) in [&mut self.cnn.layer1, &mut self.cnn.layer2, &mut self.cnn.layer3, &mut self.cnn.layer4]
            .iter_mut()
            .enumerate()
        {
            for (bi, b) in layer.iter_mut().enumerate() {
                push!(format!("l{li}b{bi}.se.fc1.w"), &mut b.se.fc1.w, &mut b.se.fc1.g_w);
                push!(format!("l{li}b{bi}.se.fc2.w"), &mut b.se.fc2.w, &mut b.se.fc2.g_w);
            }
        }
        for (li, l) in self.rnn.layers.iter_mut().enumerate() {
            push!(format!("rnn{li}.f.w_ih"), &mut l.fwd.w_ih, &mut l.fwd.g_w_ih);
            push!(format!("rnn{li}.f.b_ih"), &mut l.fwd.b_ih, &mut l.fwd.g_b_ih);
            push!(format!("rnn{li}.f.w_hh"), &mut l.fwd.w_hh, &mut l.fwd.g_w_hh);
            push!(format!("rnn{li}.f.b_hh"), &mut l.fwd.b_hh, &mut l.fwd.g_b_hh);
            push!(format!("rnn{li}.b.w_ih"), &mut l.bwd.w_ih, &mut l.bwd.g_w_ih);
            push!(format!("rnn{li}.b.b_ih"), &mut l.bwd.b_ih, &mut l.bwd.g_b_ih);
            push!(format!("rnn{li}.b.w_hh"), &mut l.bwd.w_hh, &mut l.bwd.g_w_hh);
            push!(format!("rnn{li}.b.b_hh"), &mut l.bwd.b_hh, &mut l.bwd.g_b_hh);
            push!(format!("rnn{li}.proj.w"), &mut l.proj_w, &mut l.g_proj_w);
            push!(format!("rnn{li}.proj.b"), &mut l.proj_b, &mut l.g_proj_b);
        }
        let a = &mut self.attn;
        push!("attn.i2h.w".into(), &mut a.i2h.w, &mut a.i2h.g_w);
        push!("attn.h2h.w".into(), &mut a.h2h.w, &mut a.h2h.g_w);
        push!("attn.h2h.b".into(), &mut a.h2h.b, &mut a.h2h.g_b);
        push!("attn.score.w".into(), &mut a.score.w, &mut a.score.g_w);
        push!("attn.w_ih".into(), &mut a.w_ih, &mut a.g_w_ih);
        push!("attn.b_ih".into(), &mut a.b_ih, &mut a.g_b_ih);
        push!("attn.w_hh".into(), &mut a.w_hh, &mut a.g_w_hh);
        push!("attn.b_hh".into(), &mut a.b_hh, &mut a.g_b_hh);
        push!("attn.gen.w".into(), &mut a.generator.w, &mut a.generator.g_w);
        push!("attn.gen.b".into(), &mut a.generator.b, &mut a.generator.g_b);
        push!("ctc.w".into(), &mut self.ctc.w, &mut self.ctc.g_w);
        push!("ctc.b".into(), &mut self.ctc.b, &mut self.ctc.g_b);
        v
    }
}

// ------------------------------------------------------------ checkpoint --

fn save_checkpoint(path: &str, model: &mut Model, epoch: usize, opt: &Adam) -> Result<()> {
    let mut buf: Vec<u8> = CKPT_MAGIC.to_vec();
    let mut put = |v: &Vec<f32>, buf: &mut Vec<u8>| {
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    };
    buf.extend_from_slice(&(epoch as u64).to_le_bytes());
    buf.extend_from_slice(&(opt.t as u64).to_le_bytes());
    let params = model.params();
    buf.extend_from_slice(&(params.len() as u64).to_le_bytes());
    for pr in &params {
        let slice = unsafe { std::slice::from_raw_parts(pr.p as *const f32, pr.len) };
        put(&slice.to_vec(), &mut buf);
        // BN running stats ride along with the conv params in order
    }
    for c in model.cnn.convs_mut() {
        put(&c.bn.run_mean, &mut buf);
        put(&c.bn.run_var, &mut buf);
    }
    std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")))?;
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn load_checkpoint(path: &str, model: &mut Model, opt: &mut Adam) -> Result<usize> {
    let buf = std::fs::read(path)?;
    if &buf[..8] != CKPT_MAGIC {
        return Err(anyhow!("bad checkpoint magic"));
    }
    let mut o = 8;
    let mut get_u64 = |o: &mut usize| -> u64 {
        let v = u64::from_le_bytes(buf[*o..*o + 8].try_into().unwrap());
        *o += 8;
        v
    };
    let epoch = get_u64(&mut o) as usize;
    opt.t = get_u64(&mut o) as usize;
    let params = model.params();
    let n = get_u64(&mut o) as usize;
    assert_eq!(n, params.len(), "checkpoint param count mismatch");
    for pr in &params {
        let len = get_u64(&mut o) as usize;
        assert_eq!(len, pr.len);
        let dst = unsafe { std::slice::from_raw_parts_mut(pr.p, len) };
        for i in 0..len {
            dst[i] = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
            o += 4;
        }
    }
    for c in model.cnn.convs_mut() {
        let lm = get_u64(&mut o) as usize;
        for i in 0..lm {
            c.bn.run_mean[i] = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
            o += 4;
        }
        let lv = get_u64(&mut o) as usize;
        for i in 0..lv {
            c.bn.run_var[i] = f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
            o += 4;
        }
    }
    model.cnn.requantize_all();
    Ok(epoch)
}

// ------------------------------------------------------------------ main --

fn ctc_weight_for_epoch(epoch: usize) -> f32 {
    // reference: 0.3 -> 0.03 linearly over 15 epochs
    let (init, decay, minw) = (0.3f32, 15.0f32, 0.03f32);
    let p = ((epoch as f32 - 1.0) / decay).clamp(0.0, 1.0);
    init * (1.0 - p) + minw * p
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.exp_dir)?;
    println!("🚀 TRBA (Rust + INT8 QAT) | бэкенд: {} | {}", backend_name(), device_name());

    let charset_path = args
        .charset
        .clone()
        .or_else(|| {
            let p = Path::new(&args.train_csv).parent()?.join("charset.txt");
            if p.exists() {
                Some(p.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .or_else(|| {
            let p = Path::new(&args.exp_dir).join("charset.txt");
            if p.exists() {
                Some(p.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("не найден charset.txt (положите рядом с train.csv или укажите --charset)"))?;
    let itos = load_charset(&charset_path)?;
    let stoi: std::collections::HashMap<String, usize> = itos.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    let classes = itos.len();
    let (pad, sos, eos) = (stoi["<PAD>"], stoi["<SOS>"], stoi["<EOS>"]);
    println!("📖 charset: {classes} токенов | img {}x{} | max_len {}", args.img_h, args.img_w, args.max_len);

    let (train, skip_tr) = if args.smoke > 0 {
        (Vec::new(), 0)
    } else {
        load_csv(&args.train_csv, &stoi, args.max_len)
    };
    let (val, skip_va) = match &args.val_csv {
        Some(p) if args.smoke == 0 => load_csv(p, &stoi, args.max_len),
        _ => (Vec::new(), 0),
    };
    if args.smoke == 0 {
        println!("📄 train: {} (пропущено {skip_tr}) | val: {} (пропущено {skip_va})", train.len(), val.len());
        if train.is_empty() {
            return Err(anyhow!("пустой train.csv"));
        }
    }

    let mut model = Model::new(512, args.hidden, classes, sos, eos, pad);
    let mut opt = Adam::new();
    let stream = CudaStream::new();
    let mut sc = Scratch::new();

    let start_epoch = if args.no_resume {
        1
    } else {
        let path = match &args.resume {
            Some(p) if !p.is_empty() => p.clone(),
            _ => {
                // newest checkpoint_epoch*.bin
                let mut best: Option<(usize, String)> = None;
                if let Ok(rd) = std::fs::read_dir(&args.exp_dir) {
                    for e in rd.flatten() {
                        let n = e.file_name().to_string_lossy().into_owned();
                        if let Some(rest) = n.strip_prefix("checkpoint_epoch").and_then(|r| r.strip_suffix(".bin")) {
                            if let Ok(ep) = rest.parse::<usize>() {
                                if best.as_ref().map(|(b, _)| ep > *b).unwrap_or(true) {
                                    best = Some((ep, e.path().to_string_lossy().into_owned()));
                                }
                            }
                        }
                    }
                }
                match best {
                    Some((_, p)) => p,
                    None => String::new(),
                }
            }
        };
        if path.is_empty() {
            1
        } else {
            match load_checkpoint(&path, &mut model, &mut opt) {
                Ok(ep) => {
                    println!("📂 resume: {path} (эпоха {ep})");
                    ep + 1
                }
                Err(e) => {
                    eprintln!("⚠️  resume не удался ({e}) — обучение с нуля");
                    1
                }
            }
        }
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let img_len = 3 * args.img_h * args.img_w;
    let s_img = 1.0f32 / 127.0;

    for epoch in start_epoch..=args.epochs {
        let cw = ctc_weight_for_epoch(epoch);
        let lr = args.lr * 0.5 * (1.0 + (std::f64::consts::PI * (epoch - 1) as f64 / args.epochs as f64).cos()) as f32;
        let mut order: Vec<usize> = if args.smoke > 0 {
            (0..args.smoke * args.batch).collect()
        } else {
            (0..train.len()).collect()
        };
        order.shuffle(&mut rng);

        let mut loss_sum = 0.0f64;
        let mut attn_sum = 0.0f64;
        let mut ctc_sum = 0.0f64;
        let mut batches = 0usize;
        let t0 = std::time::Instant::now();

        for chunk in order.chunks(args.batch) {
            let b = chunk.len();
            if b == 0 {
                continue;
            }
            // ---- batch ----
            let mut imgs = vec![0i8; b * img_len];
            let steps = args.max_len + 1;
            let mut text_in = vec![vec![pad; steps]; b];
            let mut target_y = vec![vec![pad; steps]; b];
            let mut ctc_targets: Vec<Vec<usize>> = Vec::new();
            let mut ok: Vec<usize> = Vec::new();
            for (bi, &idx) in chunk.iter().enumerate() {
                let (path, tokens) = if args.smoke > 0 {
                    let mut r = rand::random::<u64>();
                    let mut t = Vec::new();
                    let l = 4 + (r % 8) as usize;
                    for _ in 0..l {
                        r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
                        t.push(3 + (r as usize % (classes - 3)));
                    }
                    (String::new(), t)
                } else {
                    (train[idx].path.clone(), train[idx].tokens.clone())
                };
                let q = if args.smoke > 0 {
                    imgs[bi * img_len..(bi + 1) * img_len].iter_mut().for_each(|p| *p = rand::random::<i8>() % 64);
                    true
                } else {
                    match load_image_q(&path, args.img_h, args.img_w) {
                        Some(q) => {
                            imgs[bi * img_len..(bi + 1) * img_len].copy_from_slice(&q);
                            true
                        }
                        None => false,
                    }
                };
                if !q {
                    continue;
                }
                ok.push(bi);
                let l = tokens.len().min(args.max_len);
                text_in[bi][0] = sos;
                for i in 0..l {
                    text_in[bi][1 + i] = tokens[i];
                }
                for i in 0..l {
                    target_y[bi][i] = tokens[i];
                }
                target_y[bi][l] = eos;
                let mut ct = tokens[..l].to_vec();
                ct.push(eos);
                ctc_targets.push(ct);
            }
            // compact to successful samples
            if ok.len() != b {
                if ok.is_empty() {
                    continue;
                }
                let nb = ok.len();
                let mut imgs2 = vec![0i8; nb * img_len];
                for (ni, &bi) in ok.iter().enumerate() {
                    imgs2[ni * img_len..(ni + 1) * img_len].copy_from_slice(&imgs[bi * img_len..(bi + 1) * img_len]);
                }
                imgs = imgs2;
                text_in = ok.iter().map(|&bi| std::mem::take(&mut text_in[bi]).clone()).collect();
                let ty: Vec<Vec<usize>> = ok.iter().map(|&bi| std::mem::take(&mut target_y[bi]).clone()).collect();
                target_y = ty;
                // ctc_targets already only for ok (pushed in order)
                return_main_step(
                    &args, &mut model, &mut opt, &stream, &mut sc, &imgs, nb, &text_in, &target_y,
                    &ctc_targets, steps, cw, lr, s_img, pad, &mut loss_sum, &mut attn_sum, &mut ctc_sum,
                )?;
            } else {
                return_main_step(
                    &args, &mut model, &mut opt, &stream, &mut sc, &imgs, b, &text_in, &target_y,
                    &ctc_targets, steps, cw, lr, s_img, pad, &mut loss_sum, &mut attn_sum, &mut ctc_sum,
                )?;
            }
            batches += 1;
            if args.smoke > 0 && batches >= args.smoke {
                break;
            }
        }

        let nb_ = batches.max(1);
        println!(
            "Эпоха {epoch}/{} | {:.0}s | loss {:.4} (attn {:.4} ctc {:.4}, w_ctc {cw:.3}) | lr {:.2e}",
            args.epochs,
            t0.elapsed().as_secs_f32(),
            loss_sum / nb_ as f64,
            attn_sum / nb_ as f64,
            ctc_sum / nb_ as f64,
            lr
        );

        if args.smoke > 0 {
            println!("✅ smoke завершён");
            break;
        }

        if epoch % args.eval_every == 0 && !val.is_empty() {
            validate(&args, &mut model, &stream, &mut sc, &val, &itos, &stoi, s_img, pad, eos, cw);
        }
        let ckpt = if args.ckpt_every > 0 && epoch % args.ckpt_every == 0 {
            Some(format!("{}/checkpoint_epoch{:03}.bin", args.exp_dir, epoch))
        } else if epoch == args.epochs {
            Some(format!("{}/checkpoint_epoch{:03}.bin", args.exp_dir, epoch))
        } else {
            None
        };
        if let Some(p) = ckpt {
            save_checkpoint(&p, &mut model, epoch, &opt)?;
            println!("💾 {}", p);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn return_main_step(
    args: &Args,
    model: &mut Model,
    opt: &mut Adam,
    stream: &CudaStream,
    sc: &mut Scratch,
    imgs: &[i8],
    b: usize,
    text_in: &[Vec<usize>],
    target_y: &[Vec<usize>],
    ctc_targets: &[Vec<usize>],
    steps: usize,
    cw: f32,
    lr: f32,
    s_img: f32,
    pad: usize,
    loss_sum: &mut f64,
    attn_sum: &mut f64,
    ctc_sum: &mut f64,
) -> Result<()> {
    let classes = model.attn.classes;
    // ---------- forward ----------
    let img_buf = Arc::new(DevBuf::new(imgs.len()));
    img_buf.upload_i8(imgs, stream);
    let (feat, trace) = model.cnn.forward(&img_buf, s_img, b, true, args.img_h, args.img_w, stream, sc);
    model.cnn.update_running_stats(&trace);
    let t = trace.out.w_feat;
    let mut feat_h = vec![0.0f32; b * t * 512];
    feat.download_f32(&mut feat_h, stream);
    stream.sync();

    let (enc, rnn_tr) = model.rnn.forward(&feat_h, b, t);
    let (logits, attn_tr) = model.attn.forward_train(&enc, text_in, b, t);
    let ctc_logits = model.ctc.forward(&enc, b * t);

    // ---------- losses ----------
    // attention CE with ignore_index = PAD
    let mut count = 0usize;
    let mut attn_loss = 0.0f64;
    let mut d_logits = vec![0.0f32; b * steps * classes];
    for bi in 0..b {
        for st in 0..steps {
            let tgt = target_y[bi][st];
            if tgt == pad {
                continue;
            }
            count += 1;
            let row = &logits[(bi * steps + st) * classes..(bi * steps + st + 1) * classes];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = row.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = ex.iter().sum();
            attn_loss += (ex[tgt] / s).ln() as f64 * -1.0;
            for k in 0..classes {
                let p = ex[k] / s;
                d_logits[(bi * steps + st) * classes + k] = p - if k == tgt { 1.0 } else { 0.0 };
            }
        }
    }
    let count = count.max(1);
    let attn_loss = attn_loss / count as f64;
    let scale = (1.0 - cw) / count as f32;
    for x in d_logits.iter_mut() {
        *x *= scale;
    }
    let (ctc_loss, mut d_ctc) = ctc_loss_backward(&ctc_logits, b, t, classes, ctc_targets, 0);
    for x in d_ctc.iter_mut() {
        *x *= cw;
    }
    let total = (1.0 - cw) as f64 * attn_loss + cw as f64 * ctc_loss;
    *loss_sum += total;
    *attn_sum += attn_loss;
    *ctc_sum += ctc_loss;

    // ---------- backward ----------
    model.zero_grad();
    let d_enc_attn = model.attn.backward(&d_logits, &attn_tr);
    let d_ctc_enc = model.ctc.backward(&enc, &d_ctc, b * t);
    let mut d_enc = vec![0.0f32; b * t * model.attn.hidden];
    for i in 0..d_enc.len() {
        d_enc[i] = d_enc_attn[i] + d_ctc_enc[i];
    }
    let d_feat = model.rnn.backward(&d_enc, &rnn_tr);
    let d_feat_buf = DevBuf::new(d_feat.len() * 4);
    d_feat_buf.upload_f32(&d_feat, stream);
    model.cnn.backward(&d_feat_buf, &trace, stream, sc);
    stream.sync();

    // ---------- clip + Adam ----------
    let mut params = model.params();
    let norm = Adam::global_norm(&params);
    let clip = 5.0f64;
    if norm > clip {
        let f = (clip / (norm + 1e-6)) as f32;
        for pr in params.iter_mut() {
            let g = unsafe { std::slice::from_raw_parts_mut(pr.g, pr.len) };
            for x in g.iter_mut() {
                *x *= f;
            }
        }
    }
    opt.step(&params, lr, args.weight_decay);
    model.cnn.requantize_all();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate(
    args: &Args,
    model: &mut Model,
    stream: &CudaStream,
    sc: &mut Scratch,
    val: &[Sample],
    itos: &[String],
    _stoi: &std::collections::HashMap<String, usize>,
    s_img: f32,
    pad: usize,
    eos: usize,
    cw: f32,
) {
    let img_len = 3 * args.img_h * args.img_w;
    let b = args.batch.min(val.len()).min(8);
    let mut correct = 0usize;
    let mut n = 0usize;
    let mut cer_sum = 0.0f64;
    let mut loss = 0.0f64;
    for chunk in val.chunks(b) {
        let bb = chunk.len();
        let mut imgs = vec![0i8; bb * img_len];
        let mut refs = Vec::new();
        for (bi, s) in chunk.iter().enumerate() {
            match load_image_q(&s.path, args.img_h, args.img_w) {
                Some(q) => imgs[bi * img_len..(bi + 1) * img_len].copy_from_slice(&q),
                None => continue,
            };
            refs.push(s.tokens.clone());
        }
        if refs.len() != bb {
            continue;
        }
        let img_buf = Arc::new(DevBuf::new(imgs.len()));
        img_buf.upload_i8(&imgs, stream);
        let (feat, _tr) = model.cnn.forward(&img_buf, s_img, bb, false, args.img_h, args.img_w, stream, sc);
        let t = _tr.out.w_feat;
        let mut feat_h = vec![0.0f32; bb * t * 512];
        feat.download_f32(&mut feat_h, stream);
        stream.sync();
        let (enc, _rnn_tr) = model.rnn.forward(&feat_h, bb, t);
        let preds = model.attn.greedy_decode(&enc, bb, t, args.max_len);
        for (pred, ref_ids) in preds.iter().zip(refs.iter()) {
            let pred_s: String = pred.iter().map(|&i| token_str(itos, i)).collect();
            let ref_s: String = ref_ids.iter().map(|&i| token_str(itos, i)).collect();
            if pred_s == ref_s {
                correct += 1;
            }
            let a: Vec<char> = ref_s.chars().collect();
            let c: Vec<char> = pred_s.chars().collect();
            cer_sum += edit_distance(&a, &c) as f64 / a.len().max(1) as f64;
            n += 1;
        }
        let _ = (pad, eos, cw, loss);
    }
    if n > 0 {
        println!(
            "   📊 val: acc {:.1}% | CER {:.1}% | n={n}",
            correct as f64 / n as f64 * 100.0,
            cer_sum / n as f64 * 100.0
        );
    }
}

fn token_str(itos: &[String], i: usize) -> String {
    let s = &itos[i];
    if s.starts_with('<') {
        String::new()
    } else {
        s.clone()
    }
}
