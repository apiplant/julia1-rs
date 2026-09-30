//! Decision requests and the `validate_row` contract of `julia/data.py`.
use crate::pyjson;
use anyhow::{Result, bail};
use serde_json::Value;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// `state` is text, or a JSON object/array serialized like Python's `json.dumps`.
#[derive(Clone, Debug)]
pub enum State {
    Text(String),
    Json(Value),
}

impl State {
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::String(s) => Some(Self::Text(s.clone())),
            Value::Object(_) | Value::Array(_) => Some(Self::Json(value.clone())),
            _ => None,
        }
    }

    pub fn render(&self) -> Cow<'_, str> {
        match self {
            Self::Text(s) => Cow::Borrowed(s),
            Self::Json(v) => Cow::Owned(pyjson::dumps(v)),
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            Self::Text(s) => Value::String(s.clone()),
            Self::Json(v) => v.clone(),
        }
    }
}

impl From<&str> for State {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for State {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub state: State,
    pub question: String,
    pub options: Vec<String>,
    pub qtype: QType,
}

impl Request {
    pub fn new(state: impl Into<State>, question: impl Into<String>, options: Vec<String>, qtype: QType) -> Self {
        Self { state: state.into(), question: question.into(), options, qtype }
    }

    /// Parse and validate one legacy request object (`validate_row`).
    pub fn from_value(row: &Value, line: usize) -> Result<Self> {
        let prefix = format!("JSONL line {line}: ");
        let Some(obj) = row.as_object() else { bail!("{prefix}request must be a JSON object") };
        let state = obj.get("state").and_then(State::from_value);
        let question = obj.get("question").and_then(Value::as_str);
        let (Some(state), Some(question)) = (state, question) else {
            bail!("{prefix}state must be text/JSON and question must be text")
        };
        let options: Option<Vec<String>> = obj.get("options").and_then(Value::as_array).and_then(|items| {
            items.iter().map(|x| x.as_str().filter(|s| !s.is_empty()).map(str::to_owned)).collect()
        });
        let Some(options) = options.filter(|o| (2..=20).contains(&o.len())) else {
            bail!("{prefix}options must contain 2–20 nonempty rendered descriptions")
        };
        let qtype = match obj.get("type") {
            None => QType::Choice,
            Some(v) => match v.as_str().and_then(QType::parse) {
                Some(q) => q,
                None => bail!("{prefix}type must be choice, score, or noul"),
            },
        };
        if qtype == QType::Noul && options.len() != 2 {
            bail!("{prefix}noul options must be ordered [false, true]");
        }
        if let Some(target) = obj.get("target") {
            let ok = target.as_u64().is_some_and(|t| (t as usize) < options.len()) && !target.to_string().contains('.');
            if !ok {
                bail!("{prefix}target must index the supplied option list");
            }
        }
        if let Some(teacher) = obj.get("teacher_logits").filter(|v| !v.is_null()) {
            let ok = teacher.as_array().is_some_and(|t| {
                t.len() == options.len() && t.iter().all(|x| x.as_f64().is_some_and(f64::is_finite))
            });
            if !ok {
                bail!("{prefix}teacher logits must be finite and match option count/order");
            }
        }
        Ok(Self { state, question: question.to_owned(), options, qtype })
    }

    /// Same contract for requests built in Rust.
    pub fn validate(&self, line: usize) -> Result<()> {
        let prefix = format!("JSONL line {line}: ");
        if !(2..=20).contains(&self.options.len()) || self.options.iter().any(String::is_empty) {
            bail!("{prefix}options must contain 2–20 nonempty rendered descriptions");
        }
        if let State::Json(v) = &self.state
            && !(v.is_object() || v.is_array())
        {
            bail!("{prefix}state must be text/JSON and question must be text");
        }
        if self.qtype == QType::Noul && self.options.len() != 2 {
            bail!("{prefix}noul options must be ordered [false, true]");
        }
        Ok(())
    }
}

/// Model-card presentation rules (`julia/probabilities.py`); selection uses raw logits.
pub fn display_probabilities(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let winner = argmax(values);
    if values[winner] > 0.95 && values.iter().enumerate().all(|(i, v)| i == winner || *v < 0.045) {
        return (0..values.len()).map(|i| if i == winner { 1.0 } else { 0.0 }).collect();
    }
    let visible: Vec<f64> = values.iter().map(|&v| if v >= 0.01 { v } else { 0.0 }).collect();
    let total: f64 = visible.iter().sum();
    visible.iter().map(|v| v / total).collect()
}

/// First index of the maximum (matches Python's `max(range(n), key=...)` and torch argmax).
pub fn argmax<T: PartialOrd + Copy>(values: &[T]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Float32 softmax like `torch.softmax` on the logits row.
pub fn softmax_f32(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}
