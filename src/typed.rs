//! Named typed questions (`julia/typed.py`): `predict(state=..., questions=...)`.
use crate::engine::{Engine, Logits};
use crate::request::{QType, Request, State};
use anyhow::{Result, bail};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug)]
pub struct Answer {
    pub qtype: QType,
    /// Caller IDs (choice), "0".."n-1" (score) or ["false", "true"] (noul).
    pub keys: Vec<String>,
    /// Full softmax probabilities in key order (no display rounding).
    pub probabilities: Vec<f64>,
}

impl Answer {
    pub fn choice(&self) -> Option<&str> {
        (self.qtype == QType::Choice).then(|| self.keys[crate::request::argmax(&self.probabilities)].as_str())
    }

    /// Expected zero-based rubric index.
    pub fn score(&self) -> Option<f64> {
        (self.qtype == QType::Score).then(|| self.probabilities.iter().enumerate().map(|(i, p)| i as f64 * p).sum())
    }

    /// Probability of true.
    pub fn noul(&self) -> Option<f64> {
        (self.qtype == QType::Noul).then(|| self.probabilities[1])
    }

    pub fn max_probability(&self) -> Option<f64> {
        (self.qtype != QType::Noul).then(|| self.probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    }

    pub fn to_json(&self) -> Value {
        let probabilities: Map<String, Value> =
            self.keys.iter().cloned().zip(self.probabilities.iter().map(|&p| json!(p))).collect();
        let mut out = Map::new();
        out.insert("type".into(), json!(self.qtype.name()));
        out.insert("probabilities".into(), Value::Object(probabilities));
        match self.qtype {
            QType::Choice => out.insert("choice".into(), json!(self.choice())),
            QType::Score => out.insert("score".into(), json!(self.score())),
            QType::Noul => out.insert("noul".into(), json!(self.noul())),
        };
        if let Some(m) = self.max_probability() {
            out.insert("max_probability".into(), json!(m));
        }
        Value::Object(out)
    }
}

/// (question ID, type, answer keys) for each row of a named-question request.
pub type QuestionMeta = (String, QType, Vec<String>);

/// Answers keyed by caller question IDs, in request order.
pub type Answers = Vec<(String, Answer)>;

pub fn answers_to_json(answers: &Answers) -> Value {
    let map: Map<String, Value> = answers.iter().map(|(k, a)| (k.clone(), a.to_json())).collect();
    json!({ "answers": map })
}

/// Build the rows for a named-question request; all validation happens before inference.
pub fn typed_rows(state: &State, questions: &Value) -> Result<(Vec<Request>, Vec<QuestionMeta>)> {
    let Some(questions) = questions.as_object().filter(|q| !q.is_empty()) else {
        bail!("questions must be a nonempty mapping")
    };
    let mut rows = Vec::new();
    let mut metadata = Vec::new();
    for (qid, q) in questions {
        let Some(q) = q.as_object().filter(|_| !qid.is_empty()) else {
            bail!("Questions require nonempty string IDs and question objects")
        };
        let kind = q.get("type").and_then(Value::as_str).and_then(QType::parse);
        let criteria = q.get("criteria").filter(|c| !c.is_null());
        let (keys, labels): (Vec<String>, Vec<Value>) = match kind {
            Some(QType::Choice) => {
                let Some(c) = criteria.and_then(Value::as_object).filter(|c| c.keys().all(|k| !k.is_empty())) else {
                    bail!("Choice criteria must map nonempty IDs to descriptions")
                };
                (c.keys().cloned().collect(), c.values().cloned().collect())
            }
            Some(QType::Score) => {
                let Some(c) = criteria.and_then(Value::as_array) else { bail!("Score requires an ordered rubric") };
                ((0..c.len()).map(|i| i.to_string()).collect(), c.clone())
            }
            Some(QType::Noul) => {
                let keys = vec!["false".to_owned(), "true".to_owned()];
                let labels = match criteria {
                    None => keys.iter().map(|k| json!(k)).collect(),
                    Some(c) => match c.as_object().filter(|c| c.len() == 2 && c.contains_key("false") && c.contains_key("true")) {
                        Some(c) => vec![c["false"].clone(), c["true"].clone()],
                        None => bail!("Noul criteria must map false and true to descriptions"),
                    },
                };
                (keys, labels)
            }
            None => bail!("Unsupported question type"),
        };
        let kind = kind.unwrap();
        let row = json!({
            "state": state.to_value(),
            "question": q.get("instructions").cloned().unwrap_or(Value::Null),
            "type": kind.name(),
            "options": labels,
        });
        rows.push(Request::from_value(&row, rows.len() + 1)?);
        metadata.push((qid.clone(), kind, keys));
    }
    Ok((rows, metadata))
}

/// Full-softmax answers from per-question logits (float64 math, as in Python).
pub fn answers_from_logits(metadata: Vec<QuestionMeta>, scores: Vec<Vec<f32>>) -> Result<Answers> {
    if scores.len() != metadata.len() {
        bail!("Model returned an incorrect answer count");
    }
    let mut answers = Vec::with_capacity(metadata.len());
    for ((qid, qtype, keys), z) in metadata.into_iter().zip(scores) {
        if z.len() != keys.len() || z.iter().any(|v| !v.is_finite()) {
            bail!("Invalid model scores");
        }
        let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = z.iter().map(|&v| (v as f64 - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probabilities = exps.into_iter().map(|e| e / total).collect();
        answers.push((qid, Answer { qtype, keys, probabilities }));
    }
    Ok(answers)
}

/// Named-question prediction over any logits source.
pub fn predict_typed<E: Logits + ?Sized>(engine: &E, state: &State, questions: &Value) -> Result<Answers> {
    let (rows, metadata) = typed_rows(state, questions)?;
    answers_from_logits(metadata, engine.logits(&rows)?)
}

impl Engine {
    /// Named-question API. `questions` maps IDs to `{type, instructions, criteria}`.
    pub fn predict_typed(&self, state: &State, questions: &Value) -> Result<Answers> {
        predict_typed(self, state, questions)
    }
}
