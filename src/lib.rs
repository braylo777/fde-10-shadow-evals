//! Stage 10 — Shadow Evals on Customer Data.
//!
//! * `golden`     — build a golden dataset from historical production logs: scrub PII,
//!   drop near-duplicates, label from outcome signals, stratify by intent, split by hash.
//! * `trajectory` — grade a shadow agent's tool-call trajectories against references
//!   (required order, forbidden tools, key arguments, extra steps, final-answer facts) and
//!   decide whether it may be promoted.
//! * `judge`      — calibrate an LLM-as-a-judge against human labels: agreement, Cohen's
//!   kappa, best threshold, and length / position bias.
//! * `metric`     — compare a custom business metric between production and shadow with
//!   Wilson intervals and a two-proportion z-test.

mod api;
pub mod golden;
pub mod judge;
pub mod metric;
pub mod trajectory;

pub use api::dispatch;
use serde_json::Value;

pub const OPS: &[&str] = &["build_golden_set", "grade_trajectories", "calibrate_judge", "compare_metric"];

pub(crate) fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn handle(op: &str, i: &Value) -> Result<Value, String> {
    match op {
        "build_golden_set" => golden::build_op(i),
        "grade_trajectories" => trajectory::grade_op(i),
        "calibrate_judge" => judge::calibrate_op(i),
        "compare_metric" => metric::compare_op(i),
        _ => Err(format!("unknown op `{op}`; known: {}", OPS.join(", "))),
    }
}
