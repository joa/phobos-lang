use phobos_base::half::f32_to_f16;

use super::*;

/// One block of `quant` built from a byte pattern, for the layout checks that
/// only care that a field lands where the format says it does.
fn block(quant: Quant, fill: impl Fn(usize) -> u8) -> Vec<u8> {
    (0..quant.spec().block_bytes).map(fill).collect()
}

#[test]
fn every_format_agrees_with_its_own_storage_size() {
    for quant in Quant::ALL {
        let spec = quant.spec();
        assert!(
            spec.block.is_multiple_of(spec.scale_run),
            "{}: a block of {} does not divide into runs of {}",
            spec.name,
            spec.block,
            spec.scale_run
        );
        // Two blocks of zeros decode to zeros, whatever the format: a zero
        // scale zeroes the block, which is what fused padding relies on.
        let zeros = vec![0u8; spec.block_bytes * 2];
        let mut out = vec![1.0f32; spec.block * 2];
        (spec.dequantize)(&zeros, &mut out);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "{} padding is not zero",
            spec.name
        );
    }
}

#[test]
fn ggml_type_codes_map_onto_the_registry() {
    // The bridge has to agree with the file format's own block geometry, or a
    // tensor's byte count and its decoder would disagree.
    for code in 0..31u32 {
        let Ok(ggml) = GgmlType::from_code(code) else {
            continue;
        };
        let Some(quant) = ggml.quant() else { continue };
        let spec = quant.spec();
        assert_eq!(ggml.block_size(), spec.block, "{}", spec.name);
        assert_eq!(ggml.type_size(), spec.block_bytes, "{}", spec.name);
        assert_eq!(ggml.name(), spec.name);
    }
}

#[test]
fn q8_0_planes_invert_packing() {
    let (k, n) = (64usize, 3usize);
    let qs: Vec<i8> = (0..k * n).map(|i| (i as i32 % 255 - 127) as i8).collect();
    let scales: Vec<f32> = (0..(k / 32) * n)
        .map(|i| f16(0.125 * (i as f32 + 1.0)))
        .collect();

    let packed = pack_q8_0(&qs, &scales, k, n).unwrap();
    assert_eq!(packed.byte_len(), n * (k / 32) * 34);

    let planes = packed.planes().unwrap();
    assert_eq!(planes.qs, qs);
    assert_eq!(planes.scales, scales);

    // And the planes agree with decoding: element p of output j is
    // qs[j * k + p] scaled by the run it falls in.
    let dense = packed.dense();
    for j in 0..n {
        for p in 0..k {
            let want = f32::from(qs[j * k + p]) * scales[(p / 32) * n + j];
            assert_eq!(dense[p * n + j], want, "[{p}, {j}]");
        }
    }
}

#[test]
fn a_format_without_a_kernel_still_dequantizes() {
    for quant in Quant::ALL {
        let spec = quant.spec();
        let packed = Packed::new(quant, block(quant, |i| i as u8), spec.block, 1).unwrap();
        assert_eq!(packed.dense().len(), spec.block);
        assert_eq!(packed.has_planes(), spec.planes.is_some());
        if !packed.has_planes() {
            assert!(
                packed.planes().unwrap_err().to_string().contains(spec.name),
                "{} should name itself when it has no kernel",
                spec.name
            );
        }
    }
}

#[test]
fn stacking_appends_rows_and_zero_pads() {
    let k = 32usize;
    let one = pack_q8_0(&vec![1i8; k], &[2.0], k, 1).unwrap();
    let two = pack_q8_0(&vec![-3i8; k * 2], &[4.0, 8.0], k, 2).unwrap();

    let stacked = Packed::stack(&[&one, &two], 5).unwrap();
    assert_eq!(stacked.n(), 5);

    let planes = stacked.planes().unwrap();
    assert_eq!(&planes.qs[..k], &vec![1i8; k][..]);
    assert_eq!(&planes.qs[k..3 * k], &vec![-3i8; 2 * k][..]);
    // Scales are [k / block, n], so one run of five outputs, the last two
    // being the padding.
    assert_eq!(planes.scales, vec![2.0, 4.0, 8.0, 0.0, 0.0]);
    // dense() is [k, n], so the two padded outputs are the last two columns of
    // every row.
    let dense = stacked.dense();
    assert!(dense.chunks_exact(5).all(|row| row[3..] == [0.0, 0.0]));
}

#[test]
fn permuting_outputs_moves_whole_blocks() {
    let k = 32usize;
    let qs: Vec<i8> = (0..k * 3).map(|i| (i / k) as i8 + 1).collect();
    let packed = pack_q8_0(&qs, &[1.0, 2.0, 4.0], k, 3).unwrap();

    let moved = packed.select_outputs(&[2, 0, 1]).unwrap();
    let planes = moved.planes().unwrap();
    assert_eq!(planes.scales, vec![4.0, 1.0, 2.0]);
    assert_eq!(planes.qs[0], 3);
    assert_eq!(planes.qs[k], 1);
    assert_eq!(planes.qs[2 * k], 2);

    assert!(packed.select_outputs(&[3]).is_err());
}

#[test]
fn shapes_that_do_not_tile_are_rejected() {
    let k = 32usize;
    assert!(Packed::new(Quant::Q8_0, vec![0; 34], k, 1).is_ok());
    // One byte short, and a k that is not a whole number of blocks.
    assert!(Packed::new(Quant::Q8_0, vec![0; 33], k, 1).is_err());
    assert!(Packed::new(Quant::Q8_0, vec![0; 34], 31, 1).is_err());
    assert!(Packed::new(Quant::Q4_K, vec![0; 144], 128, 1).is_err());
}

/// Q4_K decodes `d * scale[run] * q - dmin * min[run]`, so choosing the two
/// six-bit indices and the nibble makes the expected value exact. This is the
/// packing the format is easiest to get wrong: the last four runs take their
/// top two bits from the first four runs' spare bits.
#[test]
fn q4_k_unpacks_its_six_bit_scales() {
    let (d, dmin) = (0.5f32, 0.25f32);
    let scales: [u8; 8] = [1, 9, 17, 25, 33, 41, 49, 63];
    let mins: [u8; 8] = [62, 2, 10, 18, 26, 34, 42, 50];
    // Run r takes nibble r % 15 + 1, so no two adjacent runs share a value.
    let nibble = |run: usize| (run % 15 + 1) as u8;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    bytes.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    bytes.extend_from_slice(&q4_k::pack_scales(&scales, &mins));
    for plane in 0..4 {
        let (lo, hi) = (nibble(2 * plane), nibble(2 * plane + 1));
        bytes.extend(std::iter::repeat_n(lo | (hi << 4), 32));
    }
    assert_eq!(bytes.len(), 144);

    let packed = Packed::new(Quant::Q4_K, bytes, 256, 1).unwrap();
    let mut out = vec![0.0f32; 256];
    packed.row_into(0, &mut out).unwrap();

    for run in 0..8 {
        let want =
            d * f32::from(scales[run]) * f32::from(nibble(run)) - dmin * f32::from(mins[run]);
        for (i, &got) in out[run * 32..][..32].iter().enumerate() {
            assert_eq!(got, want, "run {run} element {i}");
        }
    }
}

#[test]
fn q4_k_scale_packing_round_trips_every_index() {
    // Both fields are six bits, so the interesting cases are the ones whose
    // top two bits are set: those live in another byte.
    let scales: [u8; 8] = [0, 63, 32, 31, 63, 0, 48, 15];
    let mins: [u8; 8] = [63, 0, 31, 32, 15, 48, 0, 63];
    let packed = q4_k::pack_scales(&scales, &mins);
    for run in 0..8 {
        assert_eq!(
            q4_k::scale_min(run, &packed),
            (scales[run], mins[run]),
            "run {run}"
        );
    }
}

#[test]
fn legacy_nibble_formats_split_a_block_in_half() {
    // Q4_0, Q4_1, Q5_0 and Q5_1 all put the two nibbles of byte j at elements
    // j and j + 16, which is the thing that reads wrong at a glance.
    for (quant, header) in [
        (Quant::Q4_0, 2usize),
        (Quant::Q4_1, 4),
        (Quant::Q5_0, 6),
        (Quant::Q5_1, 8),
    ] {
        let spec = quant.spec();
        let mut bytes = vec![0u8; spec.block_bytes];
        bytes[..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // Byte zero alone is non-zero, and its two nibbles differ.
        bytes[header] = 0x21;

        let mut out = vec![0.0f32; spec.block];
        (spec.dequantize)(&bytes, &mut out);
        assert_ne!(out[0], out[16], "{}", spec.name);
        // Everything the byte did not name decodes to whatever zero means for
        // the format, so exactly two elements stand out.
        let odd = out.iter().filter(|&&v| v != out[1]).count();
        assert_eq!(odd, 2, "{}: {out:?}", spec.name);
    }
}

fn f16(v: f32) -> f32 {
    phobos_base::half::f16_to_f32(f32_to_f16(v))
}

/// The delta fold the fused IQ1_S projection rests on: a weight is
/// `dl * (g +- 1/8)`, `g` is -1, 0 or 1, so `8g +- 1` is an exact `i8` and the
/// weight is `(dl / 8)` times it. If this holds, the tensor-core path needs no
/// activation row sums and no second dot product for the delta.
#[test]
fn iq1s_folds_its_delta_into_an_exact_int8() {
    let grid = iq1s_signed_grid();
    assert_eq!(grid.len(), 2048 * 2 * 8);
    assert!(
        grid.iter().all(|&g| (-9..=9).contains(&g)),
        "a folded IQ1_S weight left the range an i8 fragment can carry"
    );

    // A block whose bytes exercise every field, dequantized by the reference
    // and rebuilt the way the kernel does it.
    let bytes = block(Quant::IQ1_S, |i| (i * 37 + 11) as u8);
    let mut reference = [0.0f32; 256];
    (Quant::IQ1_S.spec().dequantize)(&bytes, &mut reference);

    let d = phobos_base::half::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    for ib in 0..8 {
        let qh = u16::from_le_bytes([bytes[34 + 2 * ib], bytes[34 + 2 * ib + 1]]);
        let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
        let sign = usize::from(qh >> 15);
        for l in 0..4 {
            let idx = usize::from(bytes[2 + 4 * ib + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
            for y in 0..8 {
                let folded = grid[(idx * 2 + sign) * 8 + y];
                let rebuilt = (dl / 8.0) * f32::from(folded);
                let at = ib * 32 + l * 8 + y;
                assert_eq!(
                    rebuilt, reference[at],
                    "element {at} rebuilt {rebuilt} against reference {}",
                    reference[at]
                );
            }
        }
    }
}
