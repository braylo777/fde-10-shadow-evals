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

/// Exact two-sided McNemar test: `b` and `c` are the discordant pairs
/// (shadow-only hits and production-only hits). Binomial(b+c, 0.5), computed in log space.
pub fn mcnemar_p(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut lf = vec![0.0f64; n as usize + 1];
    for i in 1..=n as usize {
        lf[i] = lf[i - 1] + (i as f64).ln();
    }
    let ln_fact = |x: u64| lf[x as usize];
    let ln_n = ln_fact(n);
    let tail: f64 = (0..=k)
        .map(|i| (ln_n - ln_fact(i) - ln_fact(n - i) - n as f64 * std::f64::consts::LN_2).exp())
        .sum();
    (2.0 * tail).min(1.0)
}

/// Paired comparison: the same tickets replayed through production and shadow, joined on
/// `key`. Only pairs where the ticket is in the metric's denominator on both sides count.
fn paired(m: &Value, prod: &[Value], shadow: &[Value], key: &str) -> Result<Value, String> {
    use std::collections::BTreeMap;
    let index: BTreeMap<String, &Value> = shadow
        .iter()
        .filter_map(|r| r.get(key).map(|k| (k.to_string(), r)))
        .collect();
    if index.len() != shadow.len() {
        return Err(format!("shadow records have duplicate or missing `{key}`"));
    }
    let mut seen = std::collections::BTreeSet::new();
    let (mut both, mut b, mut c, mut n, mut unmatched) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for p in prod {
        if let Some(k) = p.get(key) {
            if !seen.insert(k.to_string()) {
                return Err(format!("duplicate production `{key}` {k}"));
            }
        }
        let Some(s) = p.get(key).and_then(|k| index.get(&k.to_string())) else {
            unmatched += 1;
            continue;
        };
        if !(matches(p, &m["denominator"]) && matches(s, &m["denominator"])) {
            continue;
        }
        n += 1;
        match (matches(p, &m["numerator"]), matches(s, &m["numerator"])) {
            (true, true) => both += 1,
            (false, true) => b += 1,
            (true, false) => c += 1,
            _ => {}
        }
    }
    if n == 0 {
        return Err("no matched pairs".into());
    }
    Ok(json!({"pairs": n, "unmatched": unmatched, "both_hit": both,
        "shadow_only": b, "production_only": c, "p_value": r3(mcnemar_p(b, c))}))
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
        let pair = match i["paired_by"].as_str() {
            Some(key) => Some(paired(m, prod, shadow, key).map_err(|e| format!("metric {name}: {e}"))?),
            None => None,
        };
        let pval = match &pair {
            Some(pr) => pr["p_value"].as_f64().unwrap_or(1.0),
            None => two_prop_p(k1, n1, k2, n2),
        };
        let better = match &pair {
            // Paired: direction comes from the discordant pairs.
            Some(pr) => {
                let (b, c) = (pr["shadow_only"].as_u64().unwrap_or(0), pr["production_only"].as_u64().unwrap_or(0));
                if higher_better { b > c } else { b < c }
            }
            None => if higher_better { p2 > p1 } else { p2 < p1 },
        };
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
            "delta": r3(p2 - p1), "p_value": r3(pval), "verdict": verdict,
            "test": if pair.is_some() { "exact McNemar (paired)" } else { "two-proportion z (independent)" },
            "paired": pair}));
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

    #[test]
    fn mcnemar_known_values() {
        assert_eq!(mcnemar_p(0, 0), 1.0);
        // b=10, c=0: 2 * 0.5^10
        assert!((mcnemar_p(10, 0) - 2.0 / 1024.0).abs() < 1e-9);
        assert!((mcnemar_p(5, 5) - 1.0).abs() < 1e-9);
        assert!(mcnemar_p(300, 200) < 0.001);
        assert!(mcnemar_p(400, 400) > 0.9);
    }

    #[test]
    fn paired_detects_what_unpaired_misses() {
        // 200 tickets: both resolve 140; shadow also fixes 12, breaks 1.
        let mut prod = Vec::new();
        let mut sh = Vec::new();
        for i in 0..200 {
            let base = i < 140;
            prod.push(json!({"id": i, "resolved": base || (152..153).contains(&i)}));
            sh.push(json!({"id": i, "resolved": base || (140..152).contains(&i)}));
        }
        let m = json!([{"name": "fcr", "numerator": [{"field": "resolved", "eq": true}], "denominator": []}]);
        let u = compare_op(&json!({"metrics": m, "production": prod, "shadow": sh})).unwrap();
        let p = compare_op(&json!({"metrics": m, "production": prod, "shadow": sh, "paired_by": "id"})).unwrap();
        assert_eq!(u["metrics"][0]["verdict"], "no significant difference");
        assert_eq!(p["metrics"][0]["verdict"], "shadow better");
        assert_eq!(p["metrics"][0]["paired"]["shadow_only"], 12);
        assert_eq!(p["metrics"][0]["paired"]["production_only"], 1);
    }
}
