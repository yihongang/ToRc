//! npucheck <model> [n=128] [layer=0] [threads=8]: the NPU smoke test, on the dense products the NPU tier takes over (a
//! DeltaNet layer's attn_qkv, attn_gate and ssm_out; an attention layer's attn_q/k/v/output). Each matrix is uploaded to
//! the NPU, run over n f32 input rows (deterministic pseudo-random, unit scale), and compared with the CPU's product
//! (Q8 inputs, as the forward pass uses) and, for the first, with an f64 product of the dequantised weights. It prints
//! the largest |difference| and the relative RMS of each against the f64 reference or the CPU, and the median time of 5
//! runs on each side (the CPU using the whole pool).
//!
//! npucheck <model> experts [layer=6] [k=8] [threads=8]: the Inkling 2 check, on one layer's k routed experts, which are
//! what that model puts on the NPU: gate and up as one graph, the CPU's SiTU, then down as a second graph, as
//! inkling2/model.rs runs them. For blocks of 1, 2, 8, 32, 128 and 256 tokens it reports the NPU's gate/up, SiTU and
//! down times against the CPU's own experts, the largest difference, and the GMAC/s each side reached (the NPU's peak
//! is tens of TOPS, so that last pair says how far its tiles are from filled). Inkling 2 reads a token at a time, so
//! the run then gives what a whole token would take: one visit per slot, all at n=1. The larger blocks are what a
//! batched prefill would give it, if the model ever grew one.
//!
//! npucheck <model> list [layer=0]: what the two above would upload, without opening the NPU.
//!
//! Needs TORC_NPU=1 for the first two, and TORC_NPU_DIR if the GenieX ggml build is not in its default folder.

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn main() {
    use kernels::npu::{Npu, configure};

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: npucheck <model> [n=128] [layer=0] [threads=8]");
        eprintln!("       npucheck <model> experts [layer=6] [k=8] [threads=8]");
        eprintln!("       npucheck <model> list [layer=0]");
        std::process::exit(2);
    }
    // the first word after the model may be a mode, or the dense mode's token count
    let mode = args.get(1).map_or("", String::as_str);
    let (experts, list) = (mode == "experts", mode == "list");
    let pos: Vec<String> = if experts || list { args[2..].to_vec() } else { args[1..].to_vec() };
    let num = |i: usize| pos.get(i).and_then(|s| s.parse::<usize>().ok());
    // after the model: the dense mode's n, layer and threads; a mode's layer, k and threads
    let (n, layer, k) = if experts {
        (128, num(0).unwrap_or(6), num(1).unwrap_or(8))
    } else if list {
        (128, num(0).unwrap_or(0), num(1).unwrap_or(8))
    } else {
        (num(0).unwrap_or(128), num(1).unwrap_or(0), 8)
    };
    let layer = layer as u32;
    let threads = num(2).unwrap_or(8);
    let threads = model::pool::init(threads).threads as i32;
    // the weights only: Model::load would start its own NPU tier (a second session) and upload every product
    let w = match model::weights::Weights::load(&args[0]) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("npucheck: {}: {e}", args[0]);
            std::process::exit(1);
        }
    };
    let arch = arch_of(&w);
    if list {
        let names: Vec<&str> = dense_names(&arch).into_iter().filter(|s| w.layer(layer, s).is_ok()).collect();
        eprintln!(
            "{arch}, layer {layer}: dense products {}",
            if names.is_empty() { "(none)".to_string() } else { names.join(", ") }
        );
        match w.file.experts.iter().find(|x| x.layer == layer) {
            Some(e) => {
                let ks = k.min(e.n_expert as usize);
                eprintln!(
                    "  {} experts ({} used a token), d_in {} width {}; the first {ks} would go up",
                    e.n_expert, e.n_used, e.d_in, e.width
                );
            }
            None => eprintln!("  layer {layer} has no routed experts"),
        }
        if arch == "inkling2_loop" {
            // the experts mode's own inputs: the SiTU and the number of visits a token makes
            match model::models::inkling2::hparams::Hp::from_file(&w.file) {
                Ok(hp) => eprintln!(
                    "  {} slots a token, SiTU beta {} / {}, the experts mode sweeps n=1, 2, 8, 32, 128 and 256",
                    hp.n_slot, hp.situ_beta, hp.situ_linear_beta
                ),
                Err(e) => eprintln!("  {e}"),
            }
        }
        return;
    }
    if experts && arch != "inkling2_loop" {
        eprintln!(
            "npucheck: experts is the Inkling 2 check; {arch} has no one-token expert blocks (use the dense mode)"
        );
        std::process::exit(2);
    }
    if !configure() {
        eprintln!("npucheck: set TORC_NPU=1");
        std::process::exit(2);
    }
    let mut npu = match Npu::create() {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!("npucheck: no NPU session (TORC_NPU, TORC_NPU_DIR)");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("npucheck: NPU not usable: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("NPU up, min_n {}, CPU threads {threads}", npu.min_tokens());
    if experts {
        experts_check(&w, &mut npu, layer, k);
    } else {
        dense_check(&w, &mut npu, &arch, n, layer);
    }
}

/// The products the dense mode checks, by architecture: the tier's own list for the Qwen family, and Inkling 2's
/// attention (which that model leaves on the CPU, but the same shapes are worth a look).
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn dense_names(arch: &str) -> Vec<&'static str> {
    if arch == "inkling2_loop" {
        vec!["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
    } else {
        vec!["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]
    }
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn arch_of(w: &model::weights::Weights) -> String {
    format::meta::Meta::parse(&w.file.metadata)
        .map(|m| m.str("general.architecture").unwrap_or("").to_string())
        .unwrap_or_default()
}

/// One expert's gate, up and down, from its pack.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn expert_mats<'a>(w: &'a model::weights::Weights, g: u32, e: u32) -> [model::moe::Mat<'a>; 3] {
    let (data, pack) = w.expert(g, e).expect("expert");
    [&pack.gate, &pack.up, &pack.down].map(|p| model::moe::Mat {
        ty: p.ty,
        rows: p.rows as usize,
        cols: p.cols as usize,
        data: p.of(data),
    })
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn rnd() -> impl FnMut() -> f32 {
    let mut seed = 0x9e3779b97f4a7c15u64;
    move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn med(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// The densest products of a layer: the original smoke test.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn dense_check(w: &model::weights::Weights, npu: &mut kernels::npu::Npu, arch: &str, n: usize, l: u32) {
    use kernels::npu::{NpuGroup, NpuMm};
    use kernels::{Q8, QK, ops as kops};
    use model::moe::{Mat, apply_group_multi};
    use std::time::Instant;

    let names = dense_names(arch);
    if !names.iter().any(|s| w.layer(l, s).is_ok()) {
        eprintln!("npucheck: none of {} is in layer {l}", names.join(", "));
        std::process::exit(1);
    }
    // the products that went up: their id, tensor and name
    let mut jobs: Vec<(i32, model::weights::TensorRef<'_>, &str)> = Vec::new();
    for name in &names {
        let Ok(t) = w.layer(l, name) else { continue };
        let (cols, rows) = t.shape();
        // SAFETY: the mmap outlives the upload below
        match unsafe { npu.weight(t.t.ty, t.data.as_ptr(), rows, cols) } {
            Some(id) => {
                eprintln!("{name}: type {} {rows} x {cols}, id {id:?}", t.t.ty);
                jobs.push((id, t, name));
            }
            // a type the tier does not upload (IQ4_NL, MXFP4, ...): reported, not run
            None => eprintln!("{name}: type {} {rows} x {cols} is not one the NPU takes, skipped", t.t.ty),
        }
    }
    if jobs.is_empty() {
        eprintln!("npucheck: none of layer {l}'s products is a type the NPU takes");
        std::process::exit(1);
    }
    let (bytes, secs) = npu.upload().expect("upload");
    eprintln!(
        "uploaded {:.1} MiB in {:.2} s; ok: {:?}",
        bytes as f64 / 1048576.0,
        secs,
        jobs.iter().map(|&(id, ..)| npu.ok(id)).collect::<Vec<_>>()
    );
    let mut rnd = rnd();
    for (k, (id, t, name)) in jobs.iter().enumerate() {
        let (cols, rows) = t.shape();
        let x: Vec<f32> = (0..n * cols).map(|_| rnd()).collect();
        // NPU: one group, one product, into y[t * rows + r]
        let mut yn = vec![0f32; n * rows];
        let mm = [NpuMm { w: *id, y: yn.as_mut_ptr(), y_stride: rows }];
        let g =
            [NpuGroup { x: x.as_ptr(), cols, n, mm: mm.as_ptr(), n_mm: 1, ffn: [-1; 3], ffn_y: std::ptr::null_mut() }];
        let mut tn = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            // SAFETY: x and yn live across the call
            assert!(unsafe { npu.run(&g) }, "npu run failed");
            tn.push(t0.elapsed().as_secs_f64());
        }
        // CPU: the forward pass's path (each row quantised to Q8, then the group product)
        let mut xq = Q8::new(n * cols / QK);
        kops::q8k_quantize(&x, xq.view_mut());
        let mut yc = vec![0f32; n * rows];
        let mut tc = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            apply_group_multi(&mut [(Mat::of(t), &mut yc[..])], xq.view(), n);
            tc.push(t0.elapsed().as_secs_f64());
        }
        let cmpf = |a: &[f32], b: &[f64]| {
            let (mut mx, mut se, mut sr) = (0f64, 0f64, 0f64);
            for (&p, &q) in a.iter().zip(b) {
                mx = mx.max((p as f64 - q).abs());
                se += (p as f64 - q).powi(2);
                sr += q * q;
            }
            (mx, (se / sr.max(1e-300)).sqrt())
        };
        let ycd: Vec<f64> = yc.iter().map(|&v| v as f64).collect();
        let (mx, rr) = cmpf(&yn, &ycd);
        eprintln!(
            "{name}: n {n}: NPU {:.2} ms vs CPU {:.2} ms ({:.2}x); NPU vs CPU max |d| {mx:.4}, rel_rms {rr:.2e}",
            med(&mut tn) * 1e3,
            med(&mut tc) * 1e3,
            med(&mut tc.clone()) / med(&mut tn.clone())
        );
        if k == 0 {
            // an f64 reference over the dequantised weights (the first 8 rows of x, to keep it quick)
            let mut wf = vec![0f32; rows * cols];
            kops::dequantize(t.t.ty, t.data, &mut wf);
            let nr = n.min(8);
            let mut yr = vec![0f64; nr * rows];
            for ti in 0..nr {
                for r in 0..rows {
                    let (wr, xr) = (&wf[r * cols..(r + 1) * cols], &x[ti * cols..(ti + 1) * cols]);
                    yr[ti * rows + r] = wr.iter().zip(xr).map(|(&a, &b)| a as f64 * b as f64).sum();
                }
            }
            let (m1, r1) = cmpf(&yn[..nr * rows], &yr);
            let (m2, r2) = cmpf(&yc[..nr * rows], &yr);
            eprintln!(
                "{name}: vs f64 (first {nr} rows): NPU max |d| {m1:.4} rel_rms {r1:.2e}; CPU max |d| {m2:.4} rel_rms {r2:.2e}"
            );
        }
    }
    let (waited, graphs) = npu.stats();
    eprintln!("npu stats: {graphs} graphs, {:.1} ms waited", waited * 1e3);
}

/// Inkling 2: one layer's k routed experts, run the way its NPU path runs them (gate and up as one graph, the CPU's
/// SiTU, down as a second graph) for blocks of 1, 2, 8 and 32 tokens, against the CPU's own experts.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn experts_check(w: &model::weights::Weights, npu: &mut kernels::npu::Npu, layer: u32, k: usize) {
    use kernels::npu::{NpuGroup, NpuMm};
    use kernels::{Q8, QK, ops as kops};
    use model::moe::{Mat, apply_group_multi};
    use std::time::Instant;

    let hp = model::models::inkling2::hparams::Hp::from_file(&w.file).expect("inkling2_loop");
    let Some(g) = w.file.experts.iter().position(|x| x.layer == layer) else {
        eprintln!("npucheck: layer {layer} has no routed experts");
        std::process::exit(1);
    };
    let rec = &w.file.experts[g];
    let (dm, width) = (rec.d_in as usize, rec.width as usize);
    let ks = k.min(rec.n_expert as usize);
    let (beta, lb) = (hp.situ_beta, hp.situ_linear_beta);
    eprintln!(
        "experts: layer {layer}: the first {ks} of {} ({} used a token), d_in {dm}, width {width}, SiTU beta {beta} / {lb}",
        rec.n_expert, rec.n_used
    );
    let mats: Vec<[Mat; 3]> = (0..ks as u32).map(|e| expert_mats(w, g as u32, e)).collect();
    // the NPU's copies, exact for IQ4_NL: kept until the upload below
    let mut conv: Vec<Vec<u8>> = Vec::new();
    for three in &mats {
        for m in three {
            conv.push(model::npu::q8_0_copy(m));
        }
    }
    let mut ids: Vec<[i32; 3]> = Vec::new();
    for j in 0..mats.len() {
        let mut id = [0i32; 3];
        for i in 0..3 {
            let m = &mats[j][i];
            // SAFETY: conv's buffers and the mmap outlive the upload
            id[i] = unsafe { npu.weight(8, conv[j * 3 + i].as_ptr(), m.rows, m.cols) }.expect("weight refused");
        }
        ids.push(id);
    }
    let (bytes, secs) = npu.upload().expect("upload");
    let ok = ids.iter().all(|i| i.iter().all(|&x| npu.ok(x)));
    eprintln!("uploaded {:.1} MiB in {:.2} s; ok: {ok}", bytes as f64 / 1048576.0, secs);
    let situ = |a: f32, b: f32| (beta * (a / beta).tanh() * model::ops::sigmoid(a)) * (lb * (b / lb).tanh());
    let mut rnd = rnd();
    // one visit at n=1, which is all Inkling 2 ever gives the experts
    let mut visit = (f64::MAX, f64::MAX);
    // one token is all Inkling 2 has; the larger blocks are what a batched prefill would give it, and the point at
    // which the NPU's tiles stop being underfilled (it runs at well under a tenth of its peak at n=32)
    for n in [1usize, 2, 8, 32, 128, 256] {
        let x: Vec<f32> = (0..n * dm).map(|_| rnd()).collect();
        // the NPU's buffers and graphs: gate and up for every expert in one, down for every expert in a second
        let mut gs = vec![0f32; ks * n * width];
        let mut us = vec![0f32; ks * n * width];
        let mut hs = vec![0f32; ks * n * width];
        let mut ds = vec![0f32; ks * n * dm];
        let mut umm: Vec<NpuMm> = Vec::with_capacity(2 * ks);
        for j in 0..ks {
            umm.push(NpuMm { w: ids[j][0], y: gs[j * n * width..].as_mut_ptr(), y_stride: 0 });
            umm.push(NpuMm { w: ids[j][1], y: us[j * n * width..].as_mut_ptr(), y_stride: 0 });
        }
        let ubase = umm.as_ptr();
        let ugroups: Vec<NpuGroup> = (0..ks)
            .map(|j| NpuGroup {
                x: x.as_ptr(),
                cols: dm,
                n,
                // SAFETY: 2 * j is inside umm
                mm: unsafe { ubase.add(2 * j) },
                n_mm: 2,
                ffn: [-1; 3],
                ffn_y: std::ptr::null_mut(),
            })
            .collect();
        let mut dmm: Vec<NpuMm> = Vec::with_capacity(ks);
        for j in 0..ks {
            dmm.push(NpuMm { w: ids[j][2], y: ds[j * n * dm..].as_mut_ptr(), y_stride: 0 });
        }
        let dbase = dmm.as_ptr();
        let dgroups: Vec<NpuGroup> = (0..ks)
            .map(|j| NpuGroup {
                x: hs[j * n * width..].as_ptr(),
                cols: width,
                n,
                // SAFETY: j is inside dmm
                mm: unsafe { dbase.add(j) },
                n_mm: 1,
                ffn: [-1; 3],
                ffn_y: std::ptr::null_mut(),
            })
            .collect();
        let mut tn = Vec::new();
        let mut t_gu = Vec::new();
        let mut t_situ = Vec::new();
        let mut t_dn = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            // SAFETY: x, gs and us live across the call
            assert!(unsafe { npu.run(&ugroups) }, "npu gate/up failed");
            t_gu.push(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            for j in 0..ks {
                let (gj, uj, hj) = (
                    &gs[j * n * width..(j + 1) * n * width],
                    &us[j * n * width..(j + 1) * n * width],
                    &mut hs[j * n * width..(j + 1) * n * width],
                );
                for r in 0..n * width {
                    hj[r] = situ(gj[r], uj[r]);
                }
            }
            t_situ.push(t1.elapsed().as_secs_f64());
            let t2 = Instant::now();
            // SAFETY: hs and ds live across the call
            assert!(unsafe { npu.run(&dgroups) }, "npu down failed");
            t_dn.push(t2.elapsed().as_secs_f64());
            tn.push(t0.elapsed().as_secs_f64());
        }
        // the CPU's own chain for each expert: Q8 input, gate and up, SiTU, Q8 hidden, down
        let mut xq = Q8::new(n * dm / QK);
        kops::q8k_quantize(&x, xq.view_mut());
        let mut gc = vec![0f32; ks * n * width];
        let mut uc = vec![0f32; ks * n * width];
        let mut hc = vec![0f32; ks * n * width];
        let mut dc = vec![0f32; ks * n * dm];
        let mut hq = Q8::new(n * width / QK);
        let mut tc = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for (j, three) in mats.iter().enumerate() {
                let (gj, uj) =
                    (&mut gc[j * n * width..(j + 1) * n * width], &mut uc[j * n * width..(j + 1) * n * width]);
                apply_group_multi(&mut [(three[0], gj), (three[1], uj)], xq.view(), n);
            }
            for j in 0..ks {
                let (gj, uj) = (&gc[j * n * width..(j + 1) * n * width], &uc[j * n * width..(j + 1) * n * width]);
                let hj = &mut hc[j * n * width..(j + 1) * n * width];
                for r in 0..n * width {
                    hj[r] = situ(gj[r], uj[r]);
                }
            }
            for (j, three) in mats.iter().enumerate() {
                kops::q8k_quantize(&hc[j * n * width..(j + 1) * n * width], hq.view_mut());
                apply_group_multi(&mut [(three[2], &mut dc[j * n * dm..(j + 1) * n * dm])], hq.view(), n);
            }
            tc.push(t0.elapsed().as_secs_f64());
        }
        // the CPU as inkling2/model.rs actually runs it: Mat::apply one token's vector at a time (it takes a single
        // vector, which is why this column exists at n=1 only), one expert after another, and its 4 MiB rule keeps a
        // 1.6 MiB expert matrix on the calling thread, so one expert is one core
        let mut solo = Vec::new();
        for _ in 0..if n == 1 { 5 } else { 0 } {
            let t0 = Instant::now();
            for (j, three) in mats.iter().enumerate() {
                let (gj, uj) =
                    (&mut gc[j * n * width..(j + 1) * n * width], &mut uc[j * n * width..(j + 1) * n * width]);
                three[0].apply(xq.view(), gj);
                three[1].apply(xq.view(), uj);
                for r in 0..n * width {
                    hc[j * n * width + r] = situ(gc[j * n * width + r], uc[j * n * width + r]);
                }
                kops::q8k_quantize(&hc[j * n * width..(j + 1) * n * width], hq.view_mut());
                three[2].apply(hq.view(), &mut dc[j * n * dm..(j + 1) * n * dm]);
            }
            solo.push(t0.elapsed().as_secs_f64());
        }
        let (mut mx, mut se, mut sr) = (0f64, 0f64, 0f64);
        for (&a, &b) in ds.iter().zip(&dc) {
            mx = mx.max((a as f64 - b as f64).abs());
            se += (a as f64 - b as f64).powi(2);
            sr += (b as f64).powi(2);
        }
        let (tn_m, tc_m) = (med(&mut tn), med(&mut tc));
        // the model's own path, not the pool's, is what the tier would replace
        let ts_m = if n == 1 {
            visit = (tn_m, med(&mut solo.clone()));
            med(&mut solo)
        } else {
            f64::NAN
        };
        // the multiply-accumulates each side did, over the median time: the NPU's peak is tens of TOPS, so this says
        // how far from filled its tiles are at this block
        let mac = 3.0 * ks as f64 * n as f64 * width as f64 * dm as f64;
        eprintln!(
            "n {n:3}: {ks} experts: NPU {:.2} ms (gate/up {:.2}, SiTU {:.2}, down {:.2}, {:.1} GMAC/s)",
            tn_m * 1e3,
            med(&mut t_gu) * 1e3,
            med(&mut t_situ) * 1e3,
            med(&mut t_dn) * 1e3,
            mac / tn_m / 1e9
        );
        if n == 1 {
            eprintln!(
                "        CPU as the model runs it (one core an expert): {:.2} ms = {:.1}% of the NPU's time; \
                 over the pool: {:.2} ms = {:.1}%; down vs CPU max |d| {mx:.4}, rel_rms {:.2e}",
                ts_m * 1e3,
                100.0 * ts_m / tn_m,
                tc_m * 1e3,
                100.0 * tc_m / tn_m,
                (se / sr.max(1e-300)).sqrt()
            );
        } else {
            eprintln!(
                "        CPU over the pool: {:.2} ms = {:.1}% of the NPU's time; down vs CPU max |d| {mx:.4}, rel_rms {:.2e}",
                tc_m * 1e3,
                100.0 * tc_m / tn_m,
                (se / sr.max(1e-300)).sqrt()
            );
        }
    }
    let slots = hp.n_slot as f64;
    eprintln!(
        "a token of {slots} visits, all at n=1: NPU experts {:.0} ms vs CPU experts {:.0} ms (the model's path) = {:.1}% of the NPU's time",
        visit.0 * 1e3 * slots,
        visit.1 * 1e3 * slots,
        100.0 * visit.1 / visit.0
    );
    let (waited, graphs) = npu.stats();
    eprintln!("npu stats: {graphs} graphs, {:.1} ms waited", waited * 1e3);
}

#[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux"))))]
fn main() {
    eprintln!("npucheck: ARM64 Windows only (the Snapdragon NPU)");
    std::process::exit(2);
}
