//! Marker serialization identical to `julia.data.sequence`, with the bounded LRU
//! caches of `FastEngine` (token fragments and full request encodings).
use crate::request::{QType, Request};
use anyhow::{Context, Result, anyhow, bail};
use lru::LruCache;
use serde_json::Value;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const OPTION_TOKENS: usize = 48;

#[derive(Clone, Debug)]
pub struct Encoded {
    pub ids: Vec<u32>,
    /// Sequence positions of the `<mask>` marker preceding each option.
    pub markers: Vec<u32>,
    pub qtype: QType,
    pub truncated: bool,
    /// Token count of each option (strict mode only, as in Python).
    pub option_tokens: Vec<usize>,
}

pub struct Encoder {
    tokenizer: Tokenizer,
    mask_token: String,
    pub mask_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
    pub max_length: usize,
    pub head_length: usize,
    pub strict: bool,
    tokens: Option<Mutex<LruCache<String, Arc<[u32]>>>>,
    encodings: Option<Mutex<LruCache<String, Arc<Encoded>>>>,
}

fn cache<V>(capacity: usize) -> Option<Mutex<LruCache<String, V>>> {
    NonZeroUsize::new(capacity).map(|c| Mutex::new(LruCache::new(c)))
}

impl Encoder {
    pub fn load(
        dir: &Path,
        max_length: usize,
        head_length: usize,
        strict: bool,
        token_cache: usize,
        encoding_cache: usize,
    ) -> Result<Self> {
        let tokenizer = std::fs::read(dir.join("tokenizer.json"))?;
        let config = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
        Self::from_parts(&tokenizer, &config, max_length, head_length, strict, token_cache, encoding_cache)
    }

    /// Like [`Encoder::load`], from the contents of `tokenizer.json` and `tokenizer_config.json`.
    pub fn from_parts(
        tokenizer_json: &[u8],
        tokenizer_config: &str,
        max_length: usize,
        head_length: usize,
        strict: bool,
        token_cache: usize,
        encoding_cache: usize,
    ) -> Result<Self> {
        if head_length + 4 >= max_length {
            bail!("max_length must leave room beyond the question head");
        }
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow!("{e}"))?;
        tokenizer.with_truncation(None).map_err(|e| anyhow!("{e}"))?;
        tokenizer.with_padding(None);
        let config: Value = serde_json::from_str(tokenizer_config)?;
        let token = |key: &str| -> Result<(String, u32)> {
            let name = config.get(key).and_then(Value::as_str).with_context(|| format!("tokenizer lacks {key}"))?;
            let id = tokenizer.token_to_id(name).with_context(|| format!("unknown {key} {name}"))?;
            Ok((name.to_owned(), id))
        };
        let (mask_token, mask_id) = token("mask_token")?;
        Ok(Self {
            mask_id,
            mask_token,
            cls_id: token("cls_token")?.1,
            sep_id: token("sep_token")?.1,
            pad_id: token("pad_token")?.1,
            tokenizer,
            max_length,
            head_length,
            strict,
            tokens: cache(token_cache),
            encodings: cache(encoding_cache),
        })
    }

    pub fn clear_cache(&self) {
        if let Some(c) = &self.tokens {
            c.lock().unwrap().clear();
        }
        if let Some(c) = &self.encodings {
            c.lock().unwrap().clear();
        }
    }

    pub fn tokenize(&self, text: &str) -> Result<Arc<[u32]>> {
        if let Some(cache) = &self.tokens
            && let Some(ids) = cache.lock().unwrap().get(text)
        {
            return Ok(ids.clone());
        }
        let encoding = self.tokenizer.encode_fast(text, false).map_err(|e| anyhow!("{e}"))?;
        let ids: Arc<[u32]> = encoding.get_ids().into();
        if let Some(cache) = &self.tokens {
            cache.lock().unwrap().put(text.to_owned(), ids.clone());
        }
        Ok(ids)
    }

    pub fn encode(&self, row: &Request) -> Result<Arc<Encoded>> {
        let state = row.state.render();
        let key = self.encodings.as_ref().map(|_| {
            let mut key = serde_json::to_string(&(&*state, &row.question, &row.options, row.qtype.name())).unwrap();
            key.push(if matches!(row.state, crate::request::State::Text(_)) { 't' } else { 'j' });
            key
        });
        if let (Some(cache), Some(key)) = (&self.encodings, &key)
            && let Some(hit) = cache.lock().unwrap().get(key)
        {
            return Ok(hit.clone());
        }
        let encoded = Arc::new(self.sequence(&state, row)?);
        if let (Some(cache), Some(key)) = (&self.encodings, key) {
            cache.lock().unwrap().put(key, encoded.clone());
        }
        Ok(encoded)
    }

    fn sequence(&self, state: &str, row: &Request) -> Result<Encoded> {
        let strict = self.strict;
        let mask = self.mask_token.as_str();
        if strict && [state, row.question.as_str()].into_iter().chain(row.options.iter().map(String::as_str)).any(|t| t.contains(mask)) {
            bail!("Reserved model marker in request");
        }
        let clean = |text: &str| text.replace(mask, " ");
        let head = self.tokenize(&format!("{} question: {}", row.qtype.name(), clean(&row.question)))?;
        let option_ids = row.options.iter().map(|x| self.tokenize(&format!(" {}", clean(x)))).collect::<Result<Vec<_>>>()?;
        if strict && option_ids.iter().any(|x| x.len() > OPTION_TOKENS) {
            bail!("Option exceeds 48-token model contract");
        }
        let mut options: Vec<Vec<u32>> = option_ids
            .iter()
            .map(|x| std::iter::once(self.mask_id).chain(x[..x.len().min(OPTION_TOKENS)].iter().copied()).collect())
            .collect();
        let head_length = self.head_length as i64;
        let mut budget = head_length - options.iter().map(Vec::len).sum::<usize>() as i64;
        if budget < 16 {
            let per_option = ((head_length - 16).div_euclid(options.len() as i64)).max(4) as usize;
            for o in &mut options {
                o.truncate(per_option);
            }
            budget = head_length - options.iter().map(Vec::len).sum::<usize>() as i64;
        }
        if strict
            && (head.len() as i64 > budget || options.iter().zip(&option_ids).any(|(x, y)| x.len() != y.len() + 1))
        {
            bail!("Question/options exceed lossless head budget");
        }
        let mut ids = Vec::with_capacity(self.max_length.min(4096));
        ids.push(self.cls_id);
        ids.extend_from_slice(&head[..head.len().min(budget.max(8) as usize)]);
        ids.push(self.sep_id);
        let mut markers = Vec::with_capacity(options.len());
        for option in &options {
            markers.push(ids.len() as u32);
            ids.extend_from_slice(option);
        }
        ids.push(self.sep_id);
        let state_ids = self.tokenize(&clean(state))?;
        let room = self.max_length as i64 - ids.len() as i64 - 1;
        if room < 1 {
            bail!("Question/options exceed sequence budget; shorten descriptions");
        }
        let room = room as usize;
        if strict && state_ids.len() > room {
            bail!("Game state exceeds lossless context budget");
        }
        ids.extend_from_slice(&state_ids[..state_ids.len().min(room)]);
        ids.push(self.sep_id);
        Ok(Encoded {
            ids,
            markers,
            qtype: row.qtype,
            truncated: state_ids.len() > room,
            option_tokens: if strict { option_ids.iter().map(|x| x.len()).collect() } else { Vec::new() },
        })
    }
}
