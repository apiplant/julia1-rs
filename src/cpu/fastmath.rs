//! Branch-free f32 exp/erf that LLVM auto-vectorizes (libm calls do not).
//! Accuracy is within a few ulp; see the tests against libm.

const ROUND_MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23

/// Cephes-style expf. Inputs are clamped to [-87.3, 88.3] (exp(-inf) -> ~1e-38).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.3);
    let shifted = x * std::f32::consts::LOG2_E + ROUND_MAGIC;
    let n = shifted - ROUND_MAGIC; // round-to-nearest(x * log2 e)
    let r = x - n * 0.693_359_4 - n * -2.121_944_4e-4;
    let mut y = 1.987_569_1e-4_f32;
    y = y * r + 1.398_2e-3;
    y = y * r + 8.333_452e-3;
    y = y * r + 4.166_579_6e-2;
    y = y * r + 1.666_666_5e-1;
    y = y * r + 5e-1;
    y = y * r * r + r + 1.0;
    let n_int = shifted.to_bits().wrapping_sub(ROUND_MAGIC.to_bits()); // two's complement n
    let scale = f32::from_bits(n_int.wrapping_add(127) << 23);
    y * scale
}

/// Rational erf approximation (Eigen/XLA float coefficients), |error| < 1e-6.
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = -2.726_142_3e-10_f32;
    p = p * x2 + 2.770_681_4e-8;
    p = p * x2 + -2.101_024e-6;
    p = p * x2 + -5.692_506_4e-5;
    p = p * x2 + -7.349_906_3e-4;
    p = p * x2 + -2.954_6e-3;
    p = p * x2 + -1.609_603_3e-2;
    let mut q = -1.456_607_2e-5_f32;
    q = q * x2 + -2.133_740_6e-4;
    q = q * x2 + -1.682_827e-3;
    q = q * x2 + -7.373_329e-3;
    q = q * x2 + -1.426_474e-2;
    x * p / q
}

/// Exact (erf-based) GELU, as `nn.GELU()` / transformers `gelu`.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_close_to_libm() {
        let mut worst = 0f32;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (a, b) = (exp(x), libm::expf(x));
            worst = worst.max(((a - b) / b).abs());
            x += 0.001_3;
        }
        assert!(worst < 3e-7, "relative error {worst}");
        assert!(exp(f32::NEG_INFINITY) < 1e-37);
    }

    #[test]
    fn erf_close_to_libm() {
        let mut worst = 0f32;
        let mut x = -6.0f32;
        while x < 6.0 {
            worst = worst.max((erf(x) - libm::erff(x)).abs());
            x += 0.000_7;
        }
        assert!(worst < 1e-6, "abs error {worst}");
    }
}
