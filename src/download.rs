//! Downloads the Julia-1 checkpoint from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/julia1-rs/Julia-1`, falling back to `~/.cache/julia1-rs/Julia-1`
//! when `$XDG_CACHE_HOME` is unset. Files are fetched to a `.part` sibling and renamed into place once
//! complete, so a killed download never leaves a checkpoint that looks done.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The Hugging Face repository the weights come from.
pub const HF_REPO: &str = "SupersonicLabs/Julia-1";

/// The files the runtime reads from a checkpoint directory.
pub const FILES: &[&str] = &[
    "julia_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
];

/// `$XDG_CACHE_HOME/julia1-rs`, defaulting to `~/.cache/julia1-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("julia1-rs"))
}

/// Where [`download`] puts (or would put) the checkpoint.
pub fn checkpoint_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(HF_REPO.rsplit('/').next().unwrap_or(HF_REPO)))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension(match dest.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut written, mut last_report) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 16 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!("\r  {name} done ({} KiB)          ", written >> 10);
    Ok(())
}

/// Downloads the checkpoint into the cache directory, skipping files already present, and returns the
/// checkpoint directory. Needs network access on first use; any failure (offline, 404, disk error, ...)
/// is returned as an error and nothing partial is left looking complete.
pub fn download() -> Result<PathBuf> {
    let dir = checkpoint_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut announced = false;
    for file in FILES {
        let dest = dir.join(file);
        if dest.is_file() {
            continue;
        }
        if !announced {
            eprintln!("Downloading {HF_REPO} (about 577 MB) from https://huggingface.co/{HF_REPO} into {}", dir.display());
            announced = true;
        }
        fetch_to_file(&format!("https://huggingface.co/{HF_REPO}/resolve/main/{file}"), &dest)?;
    }
    Ok(dir)
}
