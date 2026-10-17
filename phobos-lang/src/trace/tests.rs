use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::model::{Access, Step};
use super::*;

fn request(text: &str, params: &[(&str, i64)], dims: &[(&str, i64)], grid: Cta) -> Request {
    Request {
        name: "t".into(),
        origin: Origin::File,
        path: "t.ph".into(),
        fence: None,
        text: text.into(),
        variant: None,
        chip: "sm_75".into(),
        params: params.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        dims: dims.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        args: BTreeMap::new(),
        grid,
        commit: String::new(),
    }
}

/// The example GEMMs branch on `beta`; a nonzero one reads C.
fn reading_c(mut req: Request) -> Request {
    req.args.insert("beta".into(), 1);
    req
}

fn example(path: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    std::fs::read_to_string(root.join(path)).unwrap()
}

/// Every element of a tensor a step's accesses name, masked ones left out.
fn elements(accesses: &[Access], tensor: &str) -> BTreeSet<Vec<i64>> {
    let mut out = BTreeSet::new();
    for a in accesses
        .iter()
        .filter(|a| a.tensor.as_deref() == Some(tensor))
    {
        let mut at: Vec<Vec<i64>> = vec![Vec::new()];
        for [lo, hi] in &a.region {
            at = at
                .into_iter()
                .flat_map(|p| (*lo..*hi).map(move |i| [p.clone(), vec![i]].concat()))
                .collect();
        }
        out.extend(at);
    }
    out
}

fn steps_of<'a>(t: &'a Trace, kind: &str) -> Vec<&'a Step> {
    let ids: Vec<usize> = t
        .ops
        .iter()
        .filter(|o| o.kind == kind)
        .map(|o| o.id)
        .collect();
    t.steps.iter().filter(|s| ids.contains(&s.op)).collect()
}

/// saxpy's three slices, masked at the ragged end, against the elements
/// each CTA's slice covers by its definition.
#[test]
fn saxpy_boxes_match_a_brute_force_reference() {
    let (block, n) = (16, 60);
    let t = trace(&request(
        &example("examples/saxpy_fp32.ph"),
        &[("BLOCK", block)],
        &[("N", n)],
        [4, 1, 1],
    ))
    .unwrap();
    assert_eq!(t.ctas.len(), 4);
    for cta in 0..4 {
        let want: BTreeSet<Vec<i64>> = (cta * block..(cta + 1) * block)
            .filter(|&i| i < n)
            .map(|i| vec![i])
            .collect();
        let steps: Vec<&Step> = t.steps.iter().filter(|s| s.cta == [cta, 0, 0]).collect();
        let reads: Vec<Access> = steps.iter().flat_map(|s| s.reads.clone()).collect();
        let writes: Vec<Access> = steps.iter().flat_map(|s| s.writes.clone()).collect();
        assert_eq!(elements(&reads, "x"), want, "x in CTA {cta}");
        assert_eq!(elements(&reads, "y"), want, "y in CTA {cta}");
        assert_eq!(elements(&writes, "out"), want, "out in CTA {cta}");
        let masked = writes
            .iter()
            .find(|a| a.tensor.as_deref() == Some("out"))
            .and_then(|a| a.masked_elems);
        assert_eq!(masked, (cta == 3).then_some(4));
    }
}

/// The register matmul's k loop reads `A[pm*TM :+ TM, kt :+ TK]` and
/// `B[kt :+ TK, pn*TN :+ TN]` at every `kt`, and writes its C tile once.
#[test]
fn gemm_boxes_match_a_brute_force_reference() {
    let (tm, tn, tk, size) = (32, 32, 4, 64);
    let t = trace(&reading_c(request(
        &example("examples/gemm_fp32.ph"),
        &[("TILE_M", tm), ("TILE_N", tn), ("TILE_K", tk)],
        &[("M", size), ("N", size), ("K", size)],
        [2, 2, 1],
    )))
    .unwrap();
    for pm in 0..2 {
        for pn in 0..2 {
            let cta = [pm, pn, 0];
            let mut seen_kt = Vec::new();
            for s in steps_of(&t, "gemm_dot")
                .into_iter()
                .filter(|s| s.cta == cta)
            {
                let kt = s.iter["kt"];
                seen_kt.push(kt);
                let mut a = BTreeSet::new();
                let mut b = BTreeSet::new();
                for k in kt..kt + tk {
                    for i in pm * tm..(pm + 1) * tm {
                        a.insert(vec![i, k]);
                    }
                    for j in pn * tn..(pn + 1) * tn {
                        b.insert(vec![k, j]);
                    }
                }
                assert_eq!(elements(&s.reads, "A"), a, "A at kt {kt} in {cta:?}");
                assert_eq!(elements(&s.reads, "B"), b, "B at kt {kt} in {cta:?}");
            }
            assert_eq!(seen_kt, (0..size).step_by(tk as usize).collect::<Vec<_>>());
            let store = steps_of(&t, "gemm_store")
                .into_iter()
                .find(|s| s.cta == cta)
                .unwrap();
            let c: BTreeSet<Vec<i64>> = (pm * tm..(pm + 1) * tm)
                .flat_map(|i| (pn * tn..(pn + 1) * tn).map(move |j| vec![i, j]))
                .collect();
            assert_eq!(elements(&store.writes, "C"), c);
        }
    }
    // The pipelined staging alternates between two buffer sets, each
    // iteration ending in the loop's barrier.
    let dots = steps_of(&t, "gemm_dot");
    let set = |s: &Step| -> Vec<usize> { s.writes.iter().filter_map(|a| a.buffer).collect() };
    assert_eq!(set(dots[0]).len(), 2);
    assert_ne!(set(dots[0]), set(dots[1]));
    assert_eq!(set(dots[0]), set(dots[2]));
    assert!(
        dots.iter()
            .all(|s| s.barrier.as_ref().is_some_and(|b| b.kept))
    );
    let lp = t
        .ops
        .iter()
        .find(|o| o.kind == "for" && o.loop_info.is_some())
        .unwrap();
    assert_eq!(
        lp.loop_info.as_ref().map(|l| (l.var.as_deref(), l.trips)),
        Some((Some("kt"), 16))
    );
    let dot_op = t.ops.iter().find(|o| o.kind == "gemm_dot").unwrap();
    assert_eq!(dot_op.parent, Some(lp.id));
    let roles: BTreeMap<&str, Role> = t
        .tensors
        .iter()
        .map(|x| (x.name.as_str(), x.role))
        .collect();
    assert_eq!(roles["A"], Role::In);
    assert_eq!(roles["C"], Role::Inout);
}

const LANES: &str = "@launch(64)
@aligned(K = 128)
kernel mv(W: tensor<f32>[N, K], x: tensor<f32>[1, K], y: tensor<f32>[1, N]) {
  let row = ctx.block.x * ctx.warps + ctx.warp
  var acc: f32 = 0.0
  for k in range(0, K, ctx.lanes * 4) {
    let at = k + ctx.lane * 4
    acc += sum(W[row, at :+ 4] * x[0, at :+ 4])
  }
  let total = sum<over ctx.lane>(acc)
  if ctx.lane == 0 {
    y[0, row] = total
  }
}
";

/// The kinds `ir/uniform.rs` gives the block, the warp and the lane, and
/// what each lane of warp 0 reads and swaps.
#[test]
fn lanes_kernel_uniformity_runs_and_butterfly() {
    let t = trace(&request(LANES, &[], &[("K", 256), ("N", 4)], [2, 1, 1])).unwrap();
    let kind_of = |ir: &str| {
        t.ops
            .iter()
            .find(|o| o.ir.contains(ir))
            .unwrap()
            .uniformity
            .unwrap()
    };
    assert_eq!(kind_of("program_id"), Uniformity::Uniform);
    assert_eq!(kind_of("warp_id"), Uniformity::Warp);
    assert_eq!(kind_of("lane_id"), Uniformity::Varying);
    assert_eq!(kind_of("converged"), Uniformity::Warp);

    let loads = steps_of(&t, "lane_load");
    let first = loads.iter().find(|s| s.cta == [1, 0, 0]).unwrap();
    let lanes = first.lanes.as_ref().unwrap();
    assert_eq!(
        (lanes.vector_bytes, lanes.align_bytes),
        (Some(16), Some(16))
    );
    let runs = lanes.runs.as_ref().unwrap();
    assert_eq!(runs.len(), 32);
    for r in runs {
        // CTA 1, warp 0 owns row 2; lane l reads W[2, 4l :+ 4].
        let l = r.lane as i64;
        assert_eq!(r.access.region, vec![[2, 3], [4 * l, 4 * l + 4]]);
    }

    let fold = steps_of(&t, "collective")[0].lanes.clone().unwrap();
    let stages = fold.stages.unwrap();
    assert_eq!(stages.len(), 5);
    for (stage, m) in stages.iter().zip([16, 8, 4, 2, 1]) {
        assert_eq!(*stage, (0..32).map(|l| l ^ m).collect::<Vec<i64>>());
    }
    // Each shuffle of the fold is a step of its own with the same partners.
    let shuffles: Vec<Vec<i64>> = steps_of(&t, "shuffle")
        .iter()
        .filter(|s| s.cta == [0, 0, 0])
        .map(|s| s.lanes.as_ref().unwrap().partners.clone().unwrap())
        .collect();
    assert_eq!(shuffles, stages);
    // Only lane 0 stores.
    let store = steps_of(&t, "store")
        .into_iter()
        .find(|s| s.cta == [0, 0, 0])
        .unwrap();
    assert_eq!(elements(&store.writes, "y"), BTreeSet::from([vec![0, 0]]));
}

#[test]
fn an_address_loaded_from_memory_is_refused() {
    let src = "kernel g(I: tensor<i32>[1, 1], A: tensor<f32>[N], B: tensor<f32>[N]) {
  let i = I[0, 0]
  B[0 :+ 16] = A[i :+ 16]
}
";
    let err = trace(&request(src, &[], &[("N", 64)], [1, 1, 1])).unwrap_err();
    assert!(format!("{err:#}").contains(eval::DATA_DEPENDENT), "{err:#}");
}

#[test]
fn variants_edit_the_source_and_the_hash() {
    let src = "@launch(64)\n@aligned(K = 4)\nkernel k(A: tensor<f32>[K]) {\n}\n";
    let add = Variant {
        add: Some("@padstage".into()),
        drop: None,
    };
    let drop = |attr: &str| Variant {
        add: None,
        drop: Some(attr.into()),
    };
    assert_eq!(
        apply_variant(src, &add).unwrap(),
        "@launch(64)\n@aligned(K = 4)\n@padstage\nkernel k(A: tensor<f32>[K]) {\n}\n"
    );
    assert_eq!(
        apply_variant(src, &drop("@aligned")).unwrap(),
        "@launch(64)\nkernel k(A: tensor<f32>[K]) {\n}\n"
    );
    assert!(apply_variant(src, &drop("@pipeline")).is_err());
    assert_eq!(
        source_sha256(src, None),
        source_sha256(&src.replace('\n', "\r\n"), None)
    );
    assert_ne!(source_sha256(src, None), source_sha256(src, Some(&add)));
}

/// Checks `value` against the subset of JSON Schema the trace schema uses.
fn conforms(value: &Value, schema: &Value, root: &Value, at: &str) -> Result<(), String> {
    if let Some(r) = schema["$ref"].as_str() {
        let name = r.trim_start_matches("#/$defs/");
        return conforms(value, &root["$defs"][name], root, at);
    }
    if let Some(c) = schema.get("const")
        && value != c
    {
        return Err(format!("{at}: {value} is not {c}"));
    }
    if let Some(options) = schema["enum"].as_array()
        && !options.contains(value)
    {
        return Err(format!("{at}: {value} is none of {options:?}"));
    }
    let ok = match schema["type"].as_str() {
        Some("object") => value.is_object(),
        Some("array") => value.is_array(),
        Some("string") => value.is_string(),
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("boolean") => value.is_boolean(),
        _ => true,
    };
    if !ok {
        return Err(format!("{at}: {value} is not of type {}", schema["type"]));
    }
    if let Some(obj) = value.as_object() {
        for key in schema["required"].as_array().into_iter().flatten() {
            let key = key.as_str().unwrap();
            if !obj.contains_key(key) {
                return Err(format!("{at}: missing `{key}`"));
            }
        }
        for (key, v) in obj {
            let sub = match schema["properties"].get(key) {
                Some(sub) => sub,
                None => match &schema["additionalProperties"] {
                    Value::Bool(false) => return Err(format!("{at}: unexpected `{key}`")),
                    Value::Object(_) => &schema["additionalProperties"],
                    _ => continue,
                },
            };
            conforms(v, sub, root, &format!("{at}.{key}"))?;
        }
    }
    if let Some(items) = value.as_array() {
        let len = items.len() as u64;
        if schema["minItems"].as_u64().is_some_and(|n| len < n)
            || schema["maxItems"].as_u64().is_some_and(|n| len > n)
        {
            return Err(format!("{at}: {len} items"));
        }
        if schema.get("items").is_some() {
            for (i, item) in items.iter().enumerate() {
                conforms(item, &schema["items"], root, &format!("{at}[{i}]"))?;
            }
        }
    }
    Ok(())
}

/// The exporter's output for a tile kernel and a lanes kernel holds to
/// `website/src/data/trace.schema.json`, field for field.
#[test]
fn traces_conform_to_the_schema() {
    let schema: Value =
        serde_json::from_str(&example("website/src/data/trace.schema.json")).unwrap();
    let mut gemm = reading_c(request(
        &example("examples/gemm_fp32.ph"),
        &[("TILE_M", 32), ("TILE_N", 32), ("TILE_K", 4)],
        &[("M", 64), ("N", 64), ("K", 64)],
        [2, 2, 1],
    ));
    gemm.variant = Some(Variant {
        add: None,
        drop: Some("@aligned".into()),
    });
    let traces = [
        gemm,
        request(LANES, &[], &[("K", 256), ("N", 4)], [2, 1, 1]),
    ];
    for req in &traces {
        let value = serde_json::to_value(trace(req).unwrap()).unwrap();
        conforms(&value, &schema, &schema, "trace").unwrap();
    }
    // The checker itself rejects a stray field.
    let mut value = serde_json::to_value(trace(&traces[1]).unwrap()).unwrap();
    value["steps"][0]["extra"] = Value::Bool(true);
    assert!(conforms(&value, &schema, &schema, "trace").is_err());
}

/// A grid past the full-trace limit keeps a few CTAs in full and only the
/// tensor regions of the rest.
#[test]
fn a_large_grid_is_summarized() {
    let t = trace(&request(
        &example("examples/saxpy_fp32.ph"),
        &[("BLOCK", 16)],
        &[("N", 1600)],
        [100, 1, 1],
    ))
    .unwrap();
    assert_eq!(t.ctas.len(), FULL_CTAS);
    let last: Vec<Access> = t
        .summaries
        .iter()
        .filter(|s| s.cta == [99, 0, 0])
        .flat_map(|s| s.writes.clone())
        .collect();
    assert_eq!(elements(&last, "out").len(), 16);
    assert!(
        t.summaries
            .iter()
            .all(|s| s.reads.iter().chain(&s.writes).all(|a| a.buffer.is_none()))
    );
}

/// Each copy of an unrolled body carries its own value of the loop
/// variable, and the remainder runs the iterations a whole turn leaves.
#[test]
fn an_unrolled_loop_labels_each_copy() {
    let src = "@launch(128)
@autotune(TILE in [256], CHUNK in [2560])
kernel k(x: tensor<f32>[N], y: tensor<f32>[N]) {
  let base = ctx.block.x * CHUNK
  for<unroll 4> t in range(base, min(base + CHUNK, N), TILE) {
    y[t :+ TILE] = x[t :+ TILE] * 2.0
  }
}
";
    let t = trace(&request(src, &[], &[("N", 2560)], [1, 1, 1])).unwrap();
    let mut starts = Vec::new();
    for s in &t.steps {
        for a in s.reads.iter().chain(&s.writes).filter(|a| a.tensor.is_some()) {
            assert_eq!(a.region[0][0], s.iter["t"], "{s:?}");
        }
        if starts.last() != Some(&s.iter["t"]) {
            starts.push(s.iter["t"]);
        }
    }
    assert_eq!(starts, (0..10).map(|i| i * 256).collect::<Vec<_>>());
}
