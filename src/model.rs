//! Host copies of the inference weights (`JuliaDecisionModel` without the unused act_head).
use crate::config::ModelConfig;
use crate::weights::SafeTensors;
use anyhow::Result;

pub struct EncoderLayer {
    /// None for layer 0 (`nn.Identity`).
    pub attn_norm: Option<Vec<f32>>,
    pub wqkv: Vec<f32>,
    pub wo: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub wi: Vec<f32>,
    pub wo2: Vec<f32>,
}

/// `nn.TransformerEncoderLayer(norm_first=True, activation=relu)`.
pub struct HeadLayer {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub in_w: Vec<f32>,
    pub in_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub lin1_w: Vec<f32>,
    pub lin1_b: Vec<f32>,
    pub lin2_w: Vec<f32>,
    pub lin2_b: Vec<f32>,
}

/// LayerNorm -> Linear -> GELU -> Linear(.., 1).
pub struct Scorer {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

pub struct HostWeights {
    pub tensors: SafeTensors,
    pub emb_norm: Vec<f32>,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: Vec<f32>,
    pub type_emb: Vec<f32>,
    pub head: Vec<HeadLayer>,
    pub scorer: Scorer,
}

pub const EMBEDDING: &str = "encoder.embeddings.tok_embeddings.weight";

impl HostWeights {
    pub fn load(tensors: SafeTensors, cfg: &ModelConfig) -> Result<Self> {
        let (h, i) = (cfg.hidden, cfg.intermediate);
        let t = &tensors;
        t.f32(EMBEDDING, &[cfg.vocab, h])?;
        let layers = (0..cfg.layers)
            .map(|l| {
                let p = format!("encoder.layers.{l}.");
                Ok(EncoderLayer {
                    attn_norm: if l == 0 { None } else { Some(t.vec(&format!("{p}attn_norm.weight"), &[h])?) },
                    wqkv: t.vec(&format!("{p}attn.Wqkv.weight"), &[3 * h, h])?,
                    wo: t.vec(&format!("{p}attn.Wo.weight"), &[h, h])?,
                    mlp_norm: t.vec(&format!("{p}mlp_norm.weight"), &[h])?,
                    wi: t.vec(&format!("{p}mlp.Wi.weight"), &[2 * i, h])?,
                    wo2: t.vec(&format!("{p}mlp.Wo.weight"), &[h, i])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ff = cfg.head_ff;
        let head = (0..cfg.head_layers)
            .map(|l| {
                let p = format!("head.layers.{l}.");
                let v = |name: &str, shape: &[usize]| t.vec(&format!("{p}{name}"), shape);
                Ok(HeadLayer {
                    norm1_w: v("norm1.weight", &[h])?,
                    norm1_b: v("norm1.bias", &[h])?,
                    in_w: v("self_attn.in_proj_weight", &[3 * h, h])?,
                    in_b: v("self_attn.in_proj_bias", &[3 * h])?,
                    out_w: v("self_attn.out_proj.weight", &[h, h])?,
                    out_b: v("self_attn.out_proj.bias", &[h])?,
                    norm2_w: v("norm2.weight", &[h])?,
                    norm2_b: v("norm2.bias", &[h])?,
                    lin1_w: v("linear1.weight", &[ff, h])?,
                    lin1_b: v("linear1.bias", &[ff])?,
                    lin2_w: v("linear2.weight", &[h, ff])?,
                    lin2_b: v("linear2.bias", &[h])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let scorer = Scorer {
            norm_w: t.vec("scorer.0.weight", &[h])?,
            norm_b: t.vec("scorer.0.bias", &[h])?,
            w1: t.vec("scorer.1.weight", &[h, h])?,
            b1: t.vec("scorer.1.bias", &[h])?,
            w2: t.vec("scorer.3.weight", &[1, h])?,
            b2: t.vec("scorer.3.bias", &[1])?[0],
        };
        Ok(Self {
            emb_norm: t.vec("encoder.embeddings.norm.weight", &[h])?,
            final_norm: t.vec("encoder.final_norm.weight", &[h])?,
            type_emb: t.vec("type_emb.weight", &[3, h])?,
            layers,
            head,
            scorer,
            tensors,
        })
    }
}

/// One forward batch with no padding: sequences are packed back to back.
pub struct Batch {
    pub ids: Vec<u32>,
    /// Start row of each sequence in the packed token dimension.
    pub starts: Vec<usize>,
    pub lens: Vec<usize>,
    pub qtypes: Vec<u8>,
    /// Packed row of every marker, grouped by sequence.
    pub marker_rows: Vec<usize>,
    /// First marker index of each sequence (len = sequences + 1).
    pub marker_starts: Vec<usize>,
}

impl Batch {
    pub fn new(items: &[&crate::encode::Encoded]) -> Self {
        let tokens: usize = items.iter().map(|x| x.ids.len()).sum();
        let mut b = Batch {
            ids: Vec::with_capacity(tokens),
            starts: Vec::with_capacity(items.len()),
            lens: Vec::with_capacity(items.len()),
            qtypes: Vec::with_capacity(items.len()),
            marker_rows: Vec::new(),
            marker_starts: vec![0],
        };
        for item in items {
            let start = b.ids.len();
            b.starts.push(start);
            b.lens.push(item.ids.len());
            b.qtypes.push(item.qtype as u8);
            b.ids.extend_from_slice(&item.ids);
            b.marker_rows.extend(item.markers.iter().map(|&m| start + m as usize));
            b.marker_starts.push(b.marker_rows.len());
        }
        b
    }

    pub fn tokens(&self) -> usize {
        self.ids.len()
    }

    /// Split flat per-marker scores back into per-sequence logits.
    pub fn split_scores(&self, scores: &[f32]) -> Vec<Vec<f32>> {
        self.marker_starts.windows(2).map(|w| scores[w[0]..w[1]].to_vec()).collect()
    }
}
