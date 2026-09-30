//! Zero-copy safetensors access over a read-only memory map.
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct SafeTensors {
    mmap: Mmap,
    tensors: HashMap<String, (Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint must stay unchanged while loaded (same contract as the Python runtime).
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.starts_with(b"version https://git-lfs.github.com/spec/v1") {
            bail!("Checkpoint contains Git LFS pointers; fetch the real model weights first");
        }
        ensure!(mmap.len() >= 8, "Truncated safetensors file");
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        ensure!(8 + n <= mmap.len(), "Truncated safetensors header");
        let header: Value = serde_json::from_slice(&mmap[8..8 + n])?;
        let mut tensors = HashMap::new();
        for (name, info) in header.as_object().context("safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            ensure!(info["dtype"] == "F32", "{name}: only F32 tensors are supported");
            let shape = info["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let offsets = info["data_offsets"].as_array().context("data_offsets")?;
            let (a, b) = (offsets[0].as_u64().unwrap() as usize, offsets[1].as_u64().unwrap() as usize);
            ensure!(8 + n + b <= mmap.len() && a <= b, "{name}: data out of bounds");
            tensors.insert(name.clone(), (shape, 8 + n + a, 8 + n + b));
        }
        Ok(Self { mmap, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Borrowed when 4-byte aligned (the usual case), copied otherwise.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Cow<'_, [f32]>> {
        let (actual, a, b) = self.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        ensure!(actual == shape, "{name}: expected shape {shape:?}, found {actual:?}");
        let bytes = &self.mmap[*a..*b];
        // SAFETY: f32 has no invalid bit patterns; alignment is checked by align_to.
        let (pre, mid, post) = unsafe { bytes.align_to::<f32>() };
        if pre.is_empty() && post.is_empty() {
            return Ok(Cow::Borrowed(mid));
        }
        Ok(Cow::Owned(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()))
    }

    pub fn vec(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.f32(name, shape)?.into_owned())
    }
}
