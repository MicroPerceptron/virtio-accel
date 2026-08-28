//! P6 probes (issue #146 follow-up): pin the MMUL accumulator chain's exact rounding.
//!
//! Manual on-metal measurements — run with `--ignored --nocapture` on the reference NPU.
//! Findings so far (2026-08-28): serial chain confirmed; no persistent guard bits across an
//! X, d, -X sequence; crafted exact ties break toward zero while an organic mid-chain tie
//! broke away from zero — so the rounding is not RNE, floor, to-odd, half-away,
//! half-toward-zero, or any wider-precision single-rounding model. Guard/sticky state within
//! the mac is the open hypothesis.
#![cfg(va_xdna)]

#[path = "support/p6.rs"]
mod support;

use support::*;

#[test]
#[ignore = "manual on-metal P6 measurement"]
fn measure_accumulator_precision() {
    let Some(harness) = Harness::new() else {
        return;
    };
    // Sequence per p: chunk0 = +1.0, chunk1 = 2^-p, chunk2 = -1.0, rest zero.
    // Survives iff the accumulator's add of (1 + 2^-p) keeps the tail.
    // Structure probe: X at chunk0, -X at chunk1, then d = 3*2^-26 at chunks 2..63.
    // Serial chain: cancel first, then 62 exact d-adds -> 62 * 3 * 2^-26 = 2.7e-6.
    // Interleaved accumulators: the d's in X-carrying streams round away -> a different value.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, -64, 127);
        b.set(1, 0, 64, 127);
        for chunk in 2..CHUNKS {
            a.set(chunk, 0, 3, 120); // 3 * 2^(120+120-266) = 3 * 2^-26
            b.set(chunk, 0, 1, 120);
        }
        let c00 = harness.run_lane00(&a, &b);
        let serial = 62.0 * 3.0 * (2f64).powi(-26);
        println!("structure probe: got {c00:e}, serial-chain prediction {serial:e}");
    }

    // Tie-direction probe: acc = 1 + 3*2^-23 (odd mantissa), then +2^-24 (an exact tie
    // between mantissa 3 and 4), then -1. away-from-zero / RNE -> 4*2^-23; to-odd -> 3*2^-23.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, 3, 121);
        b.set(1, 0, 1, 122); // 3 * 2^(121+122-266) = 3*2^-23
        a.set(2, 0, 1, 121);
        b.set(2, 0, 1, 121); // 2^-24
        a.set(3, 0, -64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "tie probe: got {c00:e} — away/RNE predicts {:e}, to-odd predicts {:e}",
            4.0 * (2f32).powi(-23),
            3.0 * (2f32).powi(-23)
        );
    }
    // Negative-tie probe: same but negated X and increments; distinguishes away vs toward +inf.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, -64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, -3, 121);
        b.set(1, 0, 1, 122);
        a.set(2, 0, -1, 121);
        b.set(2, 0, 1, 121);
        a.set(3, 0, 64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "negative tie probe: got {c00:e} — away predicts {:e}, toward+inf predicts {:e}",
            -4.0 * (2f32).powi(-23),
            -3.0 * (2f32).powi(-23)
        );
    }

    // Final discriminator: acc = 1 + 2*2^-23 (EVEN mantissa), +2^-24 tie.
    // to-odd -> 3*2^-23; half-toward-zero -> 2*2^-23.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, 2, 121);
        b.set(1, 0, 1, 122);
        a.set(2, 0, 1, 121);
        b.set(2, 0, 1, 121);
        a.set(3, 0, -64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "even-base tie probe: got {c00:e} — to-odd predicts {:e}, half-toward-zero predicts {:e}",
            3.0 * (2f32).powi(-23),
            2.0 * (2f32).powi(-23)
        );
    }

    for p in 20..44 {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        // chunk0: 1.0 = (64*64) * 2^(127+127-266)
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        // chunk1: 2^-p = (1*1) * 2^(ea+eb-266), ea+eb = 266-p (split as evenly as possible)
        let ea = ((266 - p) / 2) as u8;
        let eb = (266 - p - (266 - p) / 2) as u8;
        a.set(1, 0, 1, ea);
        b.set(1, 0, 1, eb);
        // chunk2: -1.0
        a.set(2, 0, -64, 127);
        b.set(2, 0, 64, 127);
        a.negate_chunk(2); // lane 0 already -64; helper keeps intent explicit
        let c00 = harness.run_lane00(&a, &b);
        println!("p={p}: C[0,0] = {c00:e} (bits {:#010x})", c00.to_bits());
    }
}
