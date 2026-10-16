//! CTC loss and its gradient, mirroring `torch.nn.CTCLoss(blank=0,
//! reduction="mean", zero_infinity=True)` from the reference
//! (`TRBAModel.compute_ctc_loss`).
//!
//! The forward/backward DP runs on the host in f64: the CTC head is tiny
//! compared with the CNN, and keeping it in double precision removes the
//! underflow that a log-space f32 implementation fights with hacks.
//!
//! Layout: `logits` is `[T, B, C]` (time-major), which is what
//! `log_softmax(dim=2)` on `[B, T, C]` permuted by `(1, 0,2)` gives in the
//! reference.

/// Stable log-softmax over the last axis of a `[rows, C]` matrix.
pub fn log_softmax_rows(logits: &[f32], rows: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * c];
    for r in 0..rows {
        let src = &logits[r * c..(r + 1) * c];
        let dst = &mut out[r * c..(r + 1) * c];
        let mx = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut se = 0.0f64;
        for (i, &v) in src.iter().enumerate() {
            let e = ((v - mx) as f64).exp();
            dst[i] = e as f32;
            se += e;
        }
        let ls = se.ln();
        for (i, &v) in src.iter().enumerate() {
            dst[i] = (v as f64 - mx as f64 - ls) as f32;
        }
    }
    out
}

const NEG_INF: f64 = f64::NEG_INFINITY;

fn logadd(a: f64, b: f64) -> f64 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

/// Expanded target with blanks interleaved: `b l0 b l1 b ... b l_{L-1} b`,
/// length `2L+1`.  Repeated labels are allowed because a blank separates
/// every pair.
pub fn expand(target: &[usize], blank: usize) -> Vec<usize> {
    let mut e = Vec::with_capacity(2 * target.len() + 1);
    e.push(blank);
    for &t in target {
        e.push(t);
        e.push(blank);
    }
    e
}

/// Forward CTC in log space.  Returns (loss, alpha table `[T][S]`).
pub fn ctc_alpha(log_probs: &[f64], t_len: usize, c: usize, ext: &[usize]) -> (f64, Vec<f64>) {
    let s = ext.len();
    let mut alpha = vec![NEG_INF; t_len * s];
    // t = 0: only the blank and (if long enough) the first label are reachable
    alpha[0] = log_probs[ext[0]];
    if s > 1 {
        alpha[1] = log_probs[ext[1]];
    }
    for t in 1..t_len {
        let row = &log_probs[t * c..(t + 1) * c];
        for (si, &lab) in ext.iter().enumerate().take(s) {
            let mut best = alpha[(t - 1) * s + si]; // stay
            if si > 0 {
                best = logadd(best, alpha[(t - 1) * s + si - 1]); // from previous symbol
            }
            // skip-blank transition: jumping over the intermediate blank is
            // only legal when the two labels differ (otherwise the blank is
            // mandatory to separate the repeat)
            if si >= 2 && ext[si - 2] != lab {
                best = logadd(best, alpha[(t - 1) * s + si - 2]);
            }
            alpha[t * s + si] = best + row[lab];
        }
    }
    let loss = -logadd(alpha[(t_len - 1) * s + s - 1], alpha[(t_len - 1) * s + s - 2]);
    (loss, alpha)
}

/// Backward CTC in log space; returns the beta table `[T][S]`.
fn ctc_beta(log_probs: &[f64], t_len: usize, c: usize, ext: &[usize]) -> Vec<f64> {
    let s = ext.len();
    let mut beta = vec![NEG_INF; t_len * s];
    beta[(t_len - 1) * s + s - 1] = 0.0;
    if s > 1 {
        beta[(t_len - 1) * s + s - 2] = 0.0;
    }
    for t in (0..t_len - 1).rev() {
        let row = &log_probs[(t + 1) * c..(t + 2) * c];
        for (si, &lab) in ext.iter().enumerate().take(s) {
            let emit = row[lab];
            let mut best = beta[(t + 1) * s + si] + emit; // stay on same symbol
            if si + 1 < s {
                best = logadd(best, beta[(t + 1) * s + si + 1] + row[ext[si + 1]]);
            }
            if si + 2 < s && ext[si + 2] != lab {
                best = logadd(best, beta[(t + 1) * s + si + 2] + row[ext[si + 2]]);
            }
            beta[t * s + si] = best;
        }
    }
    beta
}

/// CTC loss for one batch.
///
/// * `logits`  – `[B, T, C]` row-major (as produced by the CTC head)
/// * `targets` – per-sample label id sequences (no blanks, no SOS/EOS)
/// * `blank`   – blank token id (0 in the reference)
///
/// Returns `(mean_loss, grad_logits [B, T, C])`.  PyTorch's
/// `CTCLoss(reduction="mean")` divides each sample's loss by its *target*
/// length and then averages over the batch; the gradient follows the same
/// scaling (`1 / (L_b * n)` per sample).
pub fn ctc_loss_backward(
    logits: &[f32],
    batch: usize,
    t_len: usize,
    classes: usize,
    targets: &[Vec<usize>],
    blank: usize,
) -> (f64, Vec<f32>) {
    // [B,T,C] -> log probs
    let rows = batch * t_len;
    let lp32 = log_softmax_rows(logits, rows, classes);
    let lp: Vec<f64> = lp32.iter().map(|&x| x as f64).collect();

    let mut grad = vec![0.0f32; batch * t_len * classes];
    let mut total_loss = 0.0f64;
    let counted = targets
        .iter()
        .filter(|t| !t.is_empty() && t_len >= t.len() + 1)
        .count()
        .max(1);
    let n = counted as f64;

    for b in 0..batch {
        let tgt = &targets[b];
        let l = tgt.len();
        if l == 0 || t_len < l + 1 {
            continue; // zero_infinity=True: loss becomes 0 with zero gradient
        }
        let ext = expand(tgt, blank);
        // log_probs for this sample, time-major view into [B,T,C]
        // build a [T, C] slice on the fly through index math
        let lp_b: Vec<f64> = {
            let mut v = Vec::with_capacity(t_len * classes);
            for t in 0..t_len {
                v.extend_from_slice(&lp[(b * t_len + t) * classes..(b * t_len + t + 1) * classes]);
            }
            v
        };
        let (loss, alpha) = ctc_alpha(&lp_b, t_len, classes, &ext);
        if !loss.is_finite() {
            continue; // zero_infinity
        }
        let beta = ctc_beta(&lp_b, t_len, classes, &ext);
        total_loss += loss / l as f64;

        // posterior over expanded symbols: gamma[t][s] = exp(alpha+beta+loss)
        let s = ext.len();
        let mut gamma = vec![0.0f64; t_len * classes];
        for t in 0..t_len {
            for si in 0..s {
                let g = (alpha[t * s + si] + beta[t * s + si] + loss).exp();
                gamma[t * classes + ext[si]] += g;
            }
        }
        let scale = 1.0 / (l as f64 * n);
        for t in 0..t_len {
            let prow = (b * t_len + t) * classes;
            for k in 0..classes {
                // d loss / d logit = softmax - posterior
                let p = lp32[prow + k].exp() as f64;
                grad[prow + k] = ((p - gamma[t * classes + k]) * scale) as f32;
            }
        }
    }

    (total_loss / n, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force CTC over all alignments for a tiny problem.
    #[test]
    fn ctc_matches_brute_force() {
        let c = 3usize; // a b blank
        let blank = 2;
        let t = 4usize;
        // logits chosen so probabilities are easy to enumerate
        let logits: Vec<f32> = vec![
            0.1, 0.6, 0.3, // t0
            0.4, 0.2, 0.4, // t1
            0.2, 0.5, 0.3, // t2
            0.3, 0.3, 0.4, // t3
        ];
        // probabilities
        let probs: Vec<Vec<f64>> = (0..t)
            .map(|i| {
                let row = &logits[i * c..(i + 1) * c];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f64> = row.iter().map(|&v| ((v - mx) as f64).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|&v| v / s).collect()
            })
            .collect();
        // target "ab" -> enumerating alignments: all paths collapsing to a b
        let mut total = 0.0f64;
        for p0 in 0..c {
            for p1 in 0..c {
                for p2 in 0..c {
                    for p3 in 0..c {
                        let path = [p0, p1, p2, p3];
                        let mut collapsed: Vec<usize> = Vec::new();
                        let mut prev = None;
                        for &x in &path {
                            if x != blank && prev != Some(x) {
                                collapsed.push(x);
                            }
                            prev = Some(x);
                        }
                        if collapsed == vec![0, 1] {
                            let mut p = 1.0;
                            for (i, &x) in path.iter().enumerate() {
                                p *= probs[i][x];
                            }
                            total += p;
                        }
                    }
                }
            }
        }
        // PyTorch's CTCLoss(reduction="mean") divides by the target length
        // and then by the batch size, so the reported loss is half the raw
        // -ln P for this two-character target.
        let expect = -total.ln() / 2.0;
        let (loss, _g) = ctc_loss_backward(&logits, 1, t, c, &[vec![0, 1]], blank);
        assert!((loss - expect).abs() < 1e-6, "loss {loss} vs {expect}");
    }

    /// Gradient checked against finite differences.
    #[test]
    fn ctc_gradient_finite_diff() {
        let c = 4usize;
        let t = 5usize;
        let blank = 0;
        let logits: Vec<f32> = (0..t * c).map(|i| ((i as f64 * 0.37).sin()) as f32).collect();
        let target = vec![1, 2, 1];
        let (_l, g) = ctc_loss_backward(&logits, 1, t, c, &[target.clone()], blank);
        let eps = 1e-4;
        for idx in [0usize, 3, 7, 11, 19] {
            let mut plus = logits.clone();
            let mut minus = logits.clone();
            plus[idx] += eps as f32;
            minus[idx] -= eps as f32;
            let lp = ctc_loss_backward(&plus, 1, t, c, &[target.clone()], blank).0;
            let lm = ctc_loss_backward(&minus, 1, t, c, &[target.clone()], blank).0;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - g[idx] as f64).abs() < 1e-3,
                "idx {idx}: analytic {} vs numeric {num}",
                g[idx]
            );
        }
    }
}
