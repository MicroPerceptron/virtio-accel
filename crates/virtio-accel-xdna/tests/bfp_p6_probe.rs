//! P6 probes (issue #146 follow-up): pin the MMUL accumulator chain's exact rounding.
//!
//! Manual on-metal measurements — run with `--ignored --nocapture` on the reference NPU.
//!
//! Findings so far (2026-08-28):
//! - The chunk chain is serial (structure probe), the stored accumulator is exactly FP32
//!   (X, d, -X sweep), and no guard/sticky state survives across chain steps (E1-E4 all zero).
//! - Crafted single-product exact ties break toward zero; an organic mid-chain tie broke away
//!   — so no single per-step rounding rule fits, and the divergence lives INSIDE one mac.
//! - E7/E9 rule out per-addend floor and truncation: sub-cut product tails vanish
//!   symmetrically (nearest at an internal cut), and mixed-sign 2^-30 tails cancel exactly.
//! - Model fitting against the E6 64-lane capture: per-addend-rounding models peak at 62/64;
//!   a pairwise adder-tree over the 8 products with >= 28-bit node rounding, root-added into
//!   the FP32 accumulator, reaches 63/64 across all tie-rule variants. One lane remains
//!   unexplained; pinning it (and cross-validating on fresh datasets) is the open work before
//!   the tier's oracle can model tie-adjacent accumulation.
#![cfg(va_xdna)]

#[path = "support/p6.rs"]
mod support;

use support::*;

/// E1-E4: does discarded-low-bit history (a sticky bit) persist across chain steps and
/// perturb later tie decisions?
#[test]
#[ignore = "manual on-metal P6 measurement"]
fn measure_sticky_history() {
    let Some(harness) = Harness::new() else {
        return;
    };

    // E1: X=1.0; discard t=2^-30 (sets sticky if any); tie d=2^-24; -X.
    // Persistent sticky -> tie reads above-half -> +2^-23. Clean-tie rule -> 0.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, 1, 118);
        b.set(1, 0, 1, 118); // 2^-30
        a.set(2, 0, 1, 121);
        b.set(2, 0, 1, 121); // 2^-24 tie
        a.set(3, 0, -64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "E1 discard-then-tie: got {c00:e} — sticky predicts {:e}, clean-tie predicts 0",
            (2f32).powi(-23)
        );
    }

    // E2: negative discard (X - 2^-30) then the same positive tie.
    // Value-below-stored sticky should make the tie read BELOW half -> round down -> 0.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, -1, 118);
        b.set(1, 0, 1, 118); // -2^-30
        a.set(2, 0, 1, 121);
        b.set(2, 0, 1, 121); // 2^-24 tie
        a.set(3, 0, -64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!("E2 negative-discard-then-tie: got {c00:e} (0 = reads below half or clean)");
    }

    // E3: two half-ULP adds (2^-25 twice). Guard bits would accumulate them to 2^-24;
    // sticky-only keeps the stored value at 1.0 both times.
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        a.set(1, 0, 1, 120);
        b.set(1, 0, 1, 121); // 2^-25
        a.set(2, 0, 1, 120);
        b.set(2, 0, 1, 121); // 2^-25
        a.set(3, 0, -64, 127);
        b.set(3, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "E3 two half-ULP adds: got {c00:e} — guard-bits predict {:e}, sticky-only predicts 0",
            (2f32).powi(-24)
        );
    }

    // E4: discard, then an exact intervening add (+1.0 -> 2.0), then a tie at the new scale,
    // then -2.0. Does the sticky survive an exact add?
    {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127); // 1.0
        a.set(1, 0, 1, 118);
        b.set(1, 0, 1, 118); // 2^-30 discard
        a.set(2, 0, 64, 127);
        b.set(2, 0, 64, 128); // +... = 2^(127+128-266)*4096 = 2^1 = 2.0? -> use +1.0 instead
        a.set(2, 0, 64, 127);
        b.set(2, 0, 64, 127); // +1.0 -> acc 2.0 (exact)
        a.set(3, 0, 1, 121);
        b.set(3, 0, 1, 122); // 2^-23 = tie at scale 2.0 (ULP 2^-22)
        a.set(4, 0, -64, 128);
        b.set(4, 0, 64, 127); // -2.0
        let c00 = harness.run_lane00(&a, &b);
        println!(
            "E4 discard, exact add, tie: got {c00:e} — surviving sticky predicts {:e}, else 0",
            (2f32).powi(-22)
        );
    }
}

/// E5: per-product truncation inside one mac. Chunk 0 sets acc = 1.0; chunk 1 puts the SAME
/// tiny product 2^-p in all 8 lanes (exact dot = 2^(3-p)); chunk 2 subtracts 1.0. If products
/// are summed exactly, the residue is 2^(3-p) whenever representable; if each aligned product
/// is truncated at internal width W, the residue vanishes once p exceeds W.
#[test]
#[ignore = "manual on-metal P6 measurement"]
fn measure_intra_mac_truncation() {
    let Some(harness) = Harness::new() else {
        return;
    };
    for sign in [1i8, -1i8] {
        for p in 24..34 {
            let mut a = Planes::zero();
            let mut b = Planes::zero();
            a.set(0, 0, 64, 127);
            b.set(0, 0, 64, 127);
            // 8 lanes of sign * 2^-p: split p across the two exponents.
            let ea = ((266 - p) / 2) as u8;
            let eb = (266 - p - (266 - p) / 2) as u8;
            for lane in 0..8 {
                a.set(1, lane, sign, ea);
                b.set(1, lane, 1, eb);
            }
            a.set(2, 0, -64, 127);
            b.set(2, 0, 64, 127);
            let c00 = harness.run_lane00(&a, &b);
            let exact = f64::from(sign) * (3.0 - p as f64).exp2();
            println!("E5 sign={sign} p={p}: got {c00:e}, exact-sum predicts {exact:e}");
        }
    }
}

/// E7: per-addend alignment truncation inside the mac. acc = 1.0; 61 chunks each holding
/// four +17 and four -15 products at scale 2^-30 (leading 2^-26 parts cancel; every lane
/// leaves a +2^-30 tail). Exact dot per chunk = 2^-27. Predictions after -1.0:
/// exact product sum: 61 * 2^-27 ~= 4.545e-7; per-addend truncation toward zero at ~2^-27:
/// ~1.818e-6; truncation toward -inf: 0.
#[test]
#[ignore = "manual on-metal P6 measurement"]
fn measure_intra_mac_alignment() {
    let Some(harness) = Harness::new() else {
        return;
    };
    let mut a = Planes::zero();
    let mut b = Planes::zero();
    a.set(0, 0, 64, 127);
    b.set(0, 0, 64, 127);
    for chunk in 1..62 {
        for lane in 0..4 {
            a.set(chunk, lane, 17, 118);
            b.set(chunk, lane, 1, 118); // +17 * 2^-30
        }
        for lane in 4..8 {
            a.set(chunk, lane, -15, 118);
            b.set(chunk, lane, 1, 118); // -15 * 2^-30
        }
    }
    a.set(63, 0, -64, 127);
    b.set(63, 0, 64, 127);
    let c00 = harness.run_lane00(&a, &b);
    println!(
        "E7 alignment: got {c00:e} — exact predicts {:e}, trunc-to-zero {:e}, trunc-to-neg-inf 0",
        61.0 * (2f64).powi(-27),
        61.0 * (2f64).powi(-25)
    );
}

/// E9: expose the truncation with 5 sub-cut negative tails in one mac. acc = 1.0; one chunk
/// with 5 lanes of -2^-30; -1.0. Exact model: dot = -5*2^-30, invisible -> 0. Per-addend
/// floor at cut ~2^-26..2^-27: each tail becomes -2^-c, dot ~ -5*2^-c -> rounds to -2^-24.
#[test]
#[ignore = "manual on-metal P6 measurement"]
fn measure_truncation_visibility() {
    let Some(harness) = Harness::new() else {
        return;
    };
    for lanes in [1usize, 2, 3, 5, 8] {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 127);
        b.set(0, 0, 64, 127);
        for lane in 0..lanes {
            a.set(1, lane, -1, 118);
            b.set(1, lane, 1, 118); // -2^-30 each
        }
        a.set(2, 0, -64, 127);
        b.set(2, 0, 64, 127);
        let c00 = harness.run_lane00(&a, &b);
        println!("E9 tails={lanes}: got {c00:e} (exact predicts 0)");
    }
    // Cut position: repeat with acc = 2^8 (alignment reference scales with acc).
    for lanes in [5usize, 8] {
        let mut a = Planes::zero();
        let mut b = Planes::zero();
        a.set(0, 0, 64, 131);
        b.set(0, 0, 64, 131); // 2^8
        for lane in 0..lanes {
            a.set(1, lane, -1, 118);
            b.set(1, lane, 1, 118); // -2^-30 each
        }
        a.set(2, 0, -64, 131);
        b.set(2, 0, 64, 131);
        let c00 = harness.run_lane00(&a, &b);
        println!("E9b acc=2^8 tails={lanes}: got {c00:e}");
    }
}

/// E6: capture all 64 hardware lanes for the order-contract dataset, for offline model
/// fitting (prints lane bits; compare against candidate reduction models host-side).
#[test]
#[ignore = "manual on-metal P6 measurement"]
fn capture_order_dataset_lanes() {
    let Some(harness) = Harness::new() else {
        return;
    };
    let mut state = 0x0148_1481u32;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8 as i8
    };
    let mut build = || {
        let mut units = vec![[0u8; 72]; CHUNKS];
        let mut mantissa = [[0i8; 512]; 8];
        for row in mantissa.iter_mut() {
            for lane in row.iter_mut() {
                *lane = next();
            }
        }
        for (chunk, unit) in units.iter_mut().enumerate() {
            for row in 0..8 {
                for lane in 0..8 {
                    unit[row * 8 + lane] = mantissa[row][chunk * 8 + lane] as u8;
                }
                unit[64 + row] = if chunk / 4 == 0 { 130 } else { 118 };
            }
        }
        units
    };
    let a = build();
    let b = build();
    let lanes = harness.run_all_lanes_raw(&a, &b);
    for (i, lane) in lanes.iter().enumerate() {
        println!("E6 lane {i}: {:#010x}", lane.to_bits());
    }
}

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
