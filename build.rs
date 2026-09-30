//! With feature `cuda`, compile src/cuda/kernels.cu into a fatbin with nvcc
//! (SASS for JULIA_CUDA_ARCHS plus compute_80 PTX for newer GPUs).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=JULIA_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = std::env::var("NVCC").ok().map(PathBuf::from).unwrap_or_else(|| {
        ["CUDA_HOME", "CUDA_PATH"]
            .iter()
            .filter_map(std::env::var_os)
            .map(|root| PathBuf::from(root).join("bin/nvcc"))
            .chain([PathBuf::from("/opt/cuda/bin/nvcc"), PathBuf::from("/usr/local/cuda/bin/nvcc")])
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("nvcc"))
    });
    let archs = std::env::var("JULIA_CUDA_ARCHS").unwrap_or_else(|_| "80,86,89,90,120".into());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.fatbin");
    let mut cmd = Command::new(&nvcc);
    cmd.args(["--fatbin", "-O3", "-std=c++17", "-lineinfo", "-o"]).arg(&out).arg("src/cuda/kernels.cu");
    for arch in archs.split([',', ';']).map(str::trim).filter(|a| !a.is_empty()) {
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    cmd.arg("-gencode=arch=compute_80,code=compute_80");
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed to compile src/cuda/kernels.cu");
}
