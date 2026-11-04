//! IEEE 754 binary16 ⇄ f32, for engines that take FP16 tensors (Core ML's
//! `OneComponent16Half` pixel buffers). Round to nearest, ties to even, as
//! the hardware does. Portable, so it's tested on every platform.

/// f32 → binary16 bits, rounding to nearest even. Overflow gives ±infinity,
/// NaN stays NaN.
pub fn from_f32(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): the implicit bit joins the shifted mantissa.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = u32::from(rem > mid || (rem == mid && half & 1 == 1));
        return sign | (half + up) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    // A carry out of the mantissa rolls into the exponent: correct, up to ∞.
    let up = u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1));
    sign | (half + up) as u16
}

/// binary16 bits → f32 (exact).
pub fn to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise.
            let (mut e, mut m) = (113u32, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_round_trips() {
        for h in 0..=u16::MAX {
            let f = to_f32(h);
            if f.is_nan() {
                assert!(to_f32(from_f32(f)).is_nan(), "{h:#06x}");
            } else {
                assert_eq!(from_f32(f), h, "{h:#06x} → {f}");
            }
        }
    }

    #[test]
    fn known_values_and_rounding() {
        assert_eq!(from_f32(1.0), 0x3c00);
        assert_eq!(from_f32(0.5), 0x3800);
        assert_eq!(from_f32(-2.0), 0xc000);
        assert_eq!(from_f32(65504.0), 0x7bff, "largest finite");
        assert_eq!(from_f32(65520.0), 0x7c00, "halfway to 2^16 rounds to ∞");
        assert_eq!(from_f32(2f32.powi(-24)), 0x0001, "smallest subnormal");
        assert_eq!(from_f32(2f32.powi(-25)), 0x0000, "tie rounds to even (0)");
        assert_eq!(
            from_f32(3.0 * 2f32.powi(-25)),
            0x0002,
            "tie rounds to even (2)"
        );
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(f32::INFINITY), 0x7c00);
        // 1 + 2^-11 is halfway between 1 and the next half (1 + 2^-10): even → 1.
        assert_eq!(from_f32(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(from_f32(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // Every 8-bit level survives with the error FP16 storage allows.
        for v in 0..=255u8 {
            let f = f32::from(v) / 255.0;
            assert!((to_f32(from_f32(f)) - f).abs() <= f * 2f32.powi(-11) + 1e-7);
        }
    }
}
