//! Custom business metrics, compared between production and shadow.
//!
//! A metric is a rate: numerator filter / denominator filter over records, e.g.
//! "first-contact resolution" = resolved without escalation / all tickets. Filters are lists
//! of `{field, eq | lt | gt}` conditions (all must hold). Guardrail metrics must not get worse.

use crate::r3;
use serde_json::{json, Value};

fn matches(r: &Value, conds: &Value) -> bool {
    conds
        .as_array()
        .map(|cs| {
            cs.iter().all(|c| {
                let v = &r[c["field"].as_str().unwrap_or("")];
                if let Some(eq) = c.get("eq") {
                    v == eq
                } else if let Some(lt) = c["lt"].as_f64() {
                    v.as_f64().map(|x| x < lt).unwrap_or(false)
                } else if let Some(gt) = c["gt"].as_f64() {
                    v.as_f64().map(|x| x > gt).unwrap_or(false)
                } else {
                    false
                }
            })
        })
        .unwrap_or(true)
}

/// Wilson score interval (95%).
pub fn wilson(k: f64, n: f64) -> (f64, f64) {
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let z = 1.96;
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = p + z * z / (2.0 * n);
    let m = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();
    ((c - m) / d, (c + m) / d)
}

/// Two-sided p-value for the difference of two proportions (pooled z-test).
pub fn two_prop_p(k1: f64, n1: f64, k2: f64, n2: f64) -> f64 {
    let p = (k1 + k2) / (n1 + n2);
    let se = (p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)).sqrt();
    if se == 0.0 {
        return 1.0;
    }
    let z = ((k1 / n1) - (k2 / n2)).abs() / se;
    // erfc via Abramowitz-Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * z / std::f64::consts::SQRT_2);
    let y = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (y * (-(z * z) / 2.0).exp()).clamp(0.0, 1.0)
}

fn eval(m: &Value, recs: &[Value]) -> (f64, f64) {
    let den: Vec<&Value> = recs
        .iter()
        .filter(|r| matches(r, &m["denominator"]))
        .collect();
    let num = den.iter().filter(|r| matches(r, &m["numerator"])).count();
    (num as f64, den.len() as f64)
}

pub fn compare_op(i: &Value) -> Result<Value, String> {
    let prod = i["production"]
        .as_array()
        .ok_or("missing `production` records")?;
    let shadow = i["shadow"].as_array().ok_or("missing `shadow` records")?;
    let metrics = i["metrics"].as_array().ok_or("missing `metrics`")?;
    let mut rows = Vec::new();
    let mut blocked = Vec::new();
    for m in metrics {
        let name = m["name"].as_str().unwrap_or("?");
        let higher_better = m["higher_is_better"].as_bool().unwrap_or(true);
        let (k1, n1) = eval(m, prod);
        let (k2, n2) = eval(m, shadow);
        if n1 == 0.0 || n2 == 0.0 {
            return Err(format!("metric {name}: empty denominator"));
        }
        let (p1, p2) = (k1 / n1, k2 / n2);
        let pval = two_prop_p(k1, n1, k2, n2);
        let better = if higher_better { p2 > p1 } else { p2 < p1 };
        let significant = pval < 0.05;
        let verdict = match (significant, better) {
            (true, true) => "shadow better",
            (true, false) => "shadow worse",
            _ => "no significant difference",
        };
        if m["guardrail"] == true && verdict == "shadow worse" {
            blocked.push(name.to_string());
        }
        let ci = |k, n| {
            let (a, b) = wilson(k, n);
            json!([r3(a), r3(b)])
        };
        rows.push(json!({"metric": name, "guardrail": m["guardrail"] == true,
            "production": {"rate": r3(p1), "n": n1, "ci95": ci(k1, n1)}, "shadow": {"rate": r3(p2), "n": n2, "ci95": ci(k2, n2)},
            "delta": r3(p2 - p1), "p_value": r3(pval), "verdict": verdict}));
    }
    let primary_better = rows
        .first()
        .map(|r| r["verdict"] == "shadow better")
        .unwrap_or(false);
    Ok(
        json!({"metrics": rows, "ship": primary_better && blocked.is_empty(),
        "reason": if !blocked.is_empty() { format!("guardrail regressed: {}", blocked.join(", ")) }
                  else if primary_better { "primary metric improved significantly; guardrails held".into() }
                  else { "primary metric did not improve significantly; keep collecting shadow data".into() }}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wilson_bounds() {
        let (a, b) = wilson(50.0, 100.0);
        assert!((a - 0.4038).abs() < 1e-3 && (b - 0.5962).abs() < 1e-3);
    }
    #[test]
    fn p_value_sanity() {
        assert!(two_prop_p(50.0, 100.0, 50.0, 100.0) > 0.99);
        assert!(two_prop_p(30.0, 100.0, 60.0, 100.0) < 0.001);
        // z = 1.96 -> p ≈ 0.05
        let p = two_prop_p(400.0, 1000.0, 443.0, 1000.0);
        assert!(p > 0.04 && p < 0.06, "{p}");
    }
}
