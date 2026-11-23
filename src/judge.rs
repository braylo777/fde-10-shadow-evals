//! LLM-as-a-judge calibration against human labels.
//!
//! Items: `{id, human: 0|1, judge_score: 1..5, response_len, position?}`. The judge passes an
//! item when judge_score >= threshold. We pick the threshold with the best Cohen's kappa, and
//! report bias signals that make a judge untrustworthy even when agreement looks fine.

use crate::r3;
use serde_json::{json, Value};

pub fn kappa(tp: f64, fp: f64, fn_: f64, tn: f64) -> f64 {
    let n = tp + fp + fn_ + tn;
    if n == 0.0 {
        return 0.0;
    }
    let po = (tp + tn) / n;
    let pe = ((tp + fp) * (tp + fn_) + (fn_ + tn) * (fp + tn)) / (n * n);
    if (1.0 - pe).abs() < 1e-12 {
        return if po == 1.0 { 1.0 } else { 0.0 };
    }
    (po - pe) / (1.0 - pe)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let vy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}

pub fn calibrate_op(i: &Value) -> Result<Value, String> {
    let items = i["items"].as_array().ok_or("missing `items`")?;
    if items.len() < 10 {
        return Err("need at least 10 human-labelled items to calibrate".into());
    }
    let h: Vec<bool> = items
        .iter()
        .map(|x| x["human"].as_u64() == Some(1))
        .collect();
    let s: Vec<f64> = items
        .iter()
        .map(|x| x["judge_score"].as_f64().unwrap_or(0.0))
        .collect();
    let mut sweep = Vec::new();
    let mut best = (f64::MIN, 0.0, [0.0; 4]);
    for t in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] {
        let mut m = [0.0; 4]; // tp fp fn tn
        for (hh, ss) in h.iter().zip(&s) {
            let j = *ss >= t;
            m[match (j, *hh) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            }] += 1.0;
        }
        let k = kappa(m[0], m[1], m[2], m[3]);
        sweep.push(json!({"threshold": t, "kappa": r3(k), "agreement": r3((m[0] + m[3]) / h.len() as f64)}));
        if k > best.0 {
            best = (k, t, m);
        }
    }
    let [tp, fp, fn_, tn] = best.2;
    // Length bias: does the judge score longer answers higher, beyond what humans do?
    let len: Vec<f64> = items
        .iter()
        .map(|x| x["response_len"].as_f64().unwrap_or(0.0))
        .collect();
    let hum: Vec<f64> = h.iter().map(|b| *b as u8 as f64).collect();
    let len_judge = pearson(&len, &s);
    let len_human = pearson(&len, &hum);
    // Position bias (pairwise judging): share of "A" wins when the human was indifferent or chose B.
    let pos: Vec<&Value> = items.iter().filter(|x| x["position"].is_string()).collect();
    let first_rate = if pos.is_empty() {
        Value::Null
    } else {
        json!(r3(pos.iter().filter(|x| x["position"] == "first").count()
            as f64
            / pos.len() as f64))
    };
    let verdict = match best.0 {
        k if k >= 0.8 => "almost perfect agreement: usable as a gate",
        k if k >= 0.6 => "substantial agreement: usable with spot checks",
        k if k >= 0.4 => "moderate: use for trends only, not for gating",
        _ => "poor: do not use this judge",
    };
    let mut warnings = Vec::new();
    if len_judge - len_human > 0.3 {
        warnings.push(format!(
            "length bias: judge-vs-length r={:.2} but human-vs-length r={:.2}",
            len_judge, len_human
        ));
    }
    if let Some(r) = first_rate.as_f64() {
        if r > 0.65 {
            warnings.push(format!("position bias: judge picks the first answer {:.0}% of the time; swap order and average", r * 100.0));
        }
    }
    Ok(json!({
        "best_threshold": best.1, "cohens_kappa": r3(best.0), "verdict": verdict,
        "confusion": {"judge_pass_human_pass": tp, "judge_pass_human_fail": fp, "judge_fail_human_pass": fn_, "judge_fail_human_fail": tn},
        "precision": r3(tp / (tp + fp).max(1.0)), "recall": r3(tp / (tp + fn_).max(1.0)),
        "threshold_sweep": sweep,
        "bias": {"length_corr_judge": r3(len_judge), "length_corr_human": r3(len_human), "first_position_rate": first_rate},
        "warnings": warnings
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kappa_known_values() {
        assert!((kappa(20.0, 5.0, 10.0, 15.0) - 0.4).abs() < 1e-9);
        assert_eq!(kappa(10.0, 0.0, 0.0, 10.0), 1.0);
    }
    #[test]
    fn picks_threshold_and_flags_length_bias() {
        let items: Vec<Value> = (0..20)
            .map(|k| {
                let human = (k % 2) as u64;
                let score = if human == 1 { 4.0 } else { 2.0 } + if k % 5 == 0 { 1.0 } else { 0.0 };
                json!({"human": human, "judge_score": score, "response_len": if score >= 4.0 { 900 } else { 200 } + k})
            })
            .collect();
        let r = calibrate_op(&json!({"items": items})).unwrap();
        assert!(r["cohens_kappa"].as_f64().unwrap() >= 0.8);
        assert_eq!(r["best_threshold"], 3.5);
    }
    #[test]
    fn too_few_items() {
        assert!(calibrate_op(&json!({"items": [{"human": 1, "judge_score": 5}]})).is_err());
    }
}
