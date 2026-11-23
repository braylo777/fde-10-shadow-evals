//! Golden dataset from historical logs.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

/// Mask emails and long digit runs so the golden set can be shared with the eval team.
pub fn scrub(t: &str) -> String {
    let mut out = Vec::new();
    for w in t.split(' ') {
        let core = w.trim_matches(|c: char| ",.;:()".contains(c));
        let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
        if core.contains('@') && core.contains('.') {
            out.push(w.replace(core, "[EMAIL]"));
        } else if digits >= 7 {
            out.push(w.replace(core, "[NUMBER]"));
        } else {
            out.push(w.to_string());
        }
    }
    out.join(" ")
}

fn shingles(t: &str) -> HashSet<String> {
    let words: Vec<String> = t
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    if words.len() < 3 {
        return words.into_iter().collect();
    }
    words.windows(3).map(|w| w.join(" ")).collect()
}

pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    a.intersection(b).count() as f64 / a.union(b).count() as f64
}

/// Outcome signals -> label. Accepted unedited = good; escalated / thumbs-down / heavy edit = bad.
pub fn label(r: &Value) -> &'static str {
    if r["escalated"] == true
        || r["thumbs"] == "down"
        || r["edit_ratio"].as_f64().unwrap_or(0.0) > 0.5
    {
        "bad"
    } else if r["accepted"] == true && r["edit_ratio"].as_f64().unwrap_or(0.0) <= 0.1 {
        "good"
    } else {
        "unclear"
    }
}

fn bucket(id: &str) -> u8 {
    Sha256::digest(id.as_bytes())[0] % 100
}

pub fn build_op(i: &Value) -> Result<Value, String> {
    let logs = i["logs"].as_array().ok_or("missing `logs`")?;
    let per_intent = i["per_intent"].as_u64().unwrap_or(5) as usize;
    let dup_threshold = i["near_duplicate_jaccard"].as_f64().unwrap_or(0.8);
    let holdout_pct = i["holdout_percent"].as_u64().unwrap_or(30) as u8;
    let mut kept: Vec<(Value, HashSet<String>)> = Vec::new();
    let (mut dups, mut unclear) = (0, 0);
    for r in logs {
        let prompt = scrub(r["prompt"].as_str().unwrap_or(""));
        let sh = shingles(&prompt);
        if kept.iter().any(|(_, s)| jaccard(s, &sh) >= dup_threshold) {
            dups += 1;
            continue;
        }
        let lab = label(r);
        if lab == "unclear" {
            unclear += 1;
            continue;
        }
        let id = r["id"].as_str().unwrap_or("").to_string();
        kept.push((
            json!({"id": id, "intent": r["intent"].as_str().unwrap_or("other"), "prompt": prompt,
                "reference": if lab == "good" { json!(scrub(r["response"].as_str().unwrap_or(""))) } else { Value::Null },
                "label": lab, "split": if bucket(&id) < holdout_pct { "holdout" } else { "dev" }}),
            sh,
        ));
    }
    // Stratify: up to N per intent, balancing good and bad so failures are represented.
    let mut by: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (v, _) in kept {
        by.entry(v["intent"].as_str().unwrap().to_string())
            .or_default()
            .push(v);
    }
    let mut set = Vec::new();
    let mut coverage = serde_json::Map::new();
    for (intent, mut items) in by {
        items.sort_by_key(|v| (v["label"] != "bad", bucket(v["id"].as_str().unwrap())));
        let bad: Vec<Value> = items
            .iter()
            .filter(|v| v["label"] == "bad")
            .take(per_intent / 2)
            .cloned()
            .collect();
        let good: Vec<Value> = items
            .iter()
            .filter(|v| v["label"] == "good")
            .take(per_intent - bad.len())
            .cloned()
            .collect();
        coverage.insert(intent, json!({"good": good.len(), "bad": bad.len()}));
        set.extend(bad);
        set.extend(good);
    }
    set.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Ok(json!({"golden_set": set, "coverage": coverage,
        "dropped": {"near_duplicates": dups, "unclear_outcome": unclear},
        "size": set.len(), "holdout": set.iter().filter(|v| v["split"] == "holdout").count()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scrubs_pii() {
        assert_eq!(
            scrub("Email jo@x.com about order 12345678."),
            "Email [EMAIL] about order [NUMBER]."
        );
    }
    #[test]
    fn near_duplicates_dropped() {
        let logs = json!([
            {"id": "1", "intent": "refund", "prompt": "I want a refund for my broken blender please", "accepted": true},
            {"id": "2", "intent": "refund", "prompt": "I want a refund for my broken blender please!!", "accepted": true},
            {"id": "3", "intent": "refund", "prompt": "Where is my package from last week", "escalated": true}]);
        let r = build_op(&json!({"logs": logs})).unwrap();
        assert_eq!(r["dropped"]["near_duplicates"], 1);
        assert_eq!(r["size"], 2);
    }
    #[test]
    fn split_is_deterministic() {
        assert_eq!(bucket("case-17"), bucket("case-17"));
    }
    #[test]
    fn labels() {
        assert_eq!(
            label(&json!({"accepted": true, "edit_ratio": 0.05})),
            "good"
        );
        assert_eq!(label(&json!({"accepted": true, "edit_ratio": 0.7})), "bad");
        assert_eq!(
            label(&json!({"accepted": true, "edit_ratio": 0.3})),
            "unclear"
        );
    }
}
