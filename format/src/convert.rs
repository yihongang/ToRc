//! convert.rs - a GGUF (or every part of a split one) into a ToRc file. No weight is requantised: each tensor is
//! copied, each expert's three matrices are gathered from the GGUF's 3-D expert tensors into one pack, and a
//! feed-forward whose width 256 does not divide gets zero padding (FORMAT.md section 6).

use crate::gguf::{Gguf, GgufTensor};
use crate::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

pub struct Report {
    pub file_bytes: u64,
    pub tensors: usize,
    pub moe_layers: usize,
    pub seconds: f64,
}

const EXPERT_SUFFIXES: [&str; 4] =
    ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_up_exps.weight"];

fn is_expert_tensor(name: &str) -> bool {
    EXPERT_SUFFIXES.iter().any(|s| name.ends_with(s)) && name.starts_with("blk.")
}

fn up256(n: u64) -> u64 {
    n.div_ceil(256) * 256
}

/// Whether the architecture pads gate and up itself (its forward pass widens the hidden state), so only down is
/// widened in the file.
fn pads_hidden_itself(arch: &str) -> bool {
    arch == "qwen4exp"
}

/// Whether the architecture's embedding width is not a whole number of 256-number blocks (gpt-oss: 2880), so every
/// quantised matrix that reads it gets zero padding at the end of each row, up to the next 256 (the forward pass pads
/// its activations with zeros to match, so the products are unchanged).
fn pads_input(arch: &str) -> bool {
    arch == "gpt-oss"
}

/// For a resident feed-forward matrix (a dense branch's, or a shared expert's) with a width 256 does not divide:
/// (new dims, change bit). Gate and up get zero rows (not for an architecture that pads its hidden state itself), and
/// down zero padding at the end of every row. With pads_input, any quantised matrix whose rows 256 does not divide
/// gets that padding.
fn widened_dense(arch: &str, t: &GgufTensor) -> Option<([u64; 4], u16)> {
    if t.n_dims != 2 {
        return None;
    }
    if pads_input(arch) && t.ty != 0 && !t.dims[0].is_multiple_of(256) {
        let mut d = t.dims;
        d[0] = up256(d[0]);
        return Some((d, ROW_TAILS_ADDED));
    }
    let stem = t.name.strip_prefix("blk.").and_then(|s| s.split_once('.')).map(|(_, r)| r)?;
    let mut d = t.dims;
    match stem {
        "ffn_gate.weight" | "ffn_up.weight" | "ffn_gate_shexp.weight" | "ffn_up_shexp.weight"
            if !d[1].is_multiple_of(256) && !pads_hidden_itself(arch) =>
        {
            d[1] = up256(d[1]);
            Some((d, ROWS_ADDED))
        }
        "ffn_down.weight" | "ffn_down_shexp.weight" if !d[0].is_multiple_of(256) => {
            d[0] = up256(d[0]);
            Some((d, ROW_TAILS_ADDED))
        }
        _ => None,
    }
}

/// Copies rows of `src_rb` bytes into rows of `dst_rb` bytes (dst arrives zeroed, so each row's tail stays zero).
fn copy_row_span(src: &[u8], src_rb: usize, dst: &mut [u8], dst_rb: usize) {
    for (s, d) in src.chunks_exact(src_rb).zip(dst.chunks_exact_mut(dst_rb)) {
        d[..src_rb].copy_from_slice(s);
    }
}

/// Where a matrix comes from: its origin (for an expert's piece, already moved to that expert), its type, the source's
/// bytes and row length, and the converted row length.
#[derive(Clone, Copy)]
pub struct SourceMatrix {
    pub origin: Origin,
    pub ty: u32,
    pub src_bytes: u64,
    pub src_cols: u64,
    pub dst_cols: u64,
}

/// Fills `dst` (zeroed) with bytes at..at + dst.len() of the converted matrix, read from its source in `src`. Widened
/// rows are read a whole row at a time, so `at` and the length must be whole converted rows then. Used by the
/// converter, and by the check against the source (which compares the file with what this produces).
pub fn source_range(src: &mut File, m: &SourceMatrix, at: u64, dst: &mut [u8], tmp: &mut Vec<u8>) -> Result<()> {
    if m.origin.changes & ROW_TAILS_ADDED != 0 {
        let (s, d) = (matrix_bytes(m.ty, 1, m.src_cols), matrix_bytes(m.ty, 1, m.dst_cols));
        let (Some(srb), Some(drb)) = (s, d) else { return err("widened rows are not whole blocks") };
        if !at.is_multiple_of(drb) || !(dst.len() as u64).is_multiple_of(drb) {
            return err("a widened matrix read off a row boundary");
        }
        // rows past the source's own (zero rows added as well) stay zero
        let (r0, rows) = (at / drb, dst.len() as u64 / drb);
        let rows = rows.min((m.src_bytes / srb).saturating_sub(r0));
        tmp.resize((rows * srb) as usize, 0);
        fetch(src, m.origin.offset + r0 * srb, tmp)?;
        copy_row_span(tmp, srb as usize, &mut dst[..(rows * drb) as usize], drb as usize);
    } else {
        // a copy, or zero rows added after the source's: the source's bytes from the same place, zeros past its end
        let have = m.src_bytes.saturating_sub(at).min(dst.len() as u64) as usize;
        fetch(src, m.origin.offset + at, &mut dst[..have])?;
    }
    Ok(())
}

/// One expert matrix as it sits in a pack, from its GGUF tensor: (piece without its place, source bytes per expert,
/// source row length).
fn piece_from(
    g: &GgufTensor,
    rows: u64,
    cols: u64,
    src_rows: u64,
    src_cols: u64,
    half: u16,
) -> Result<(Piece, u64, u64)> {
    let bad = || Error(format!("{}: bad type or shape", g.name));
    let src_bytes = matrix_bytes(g.ty, src_rows, src_cols).ok_or_else(bad)?;
    let stride = matrix_bytes(g.ty, g.dims[1], g.dims[0]).ok_or_else(bad)?;
    let mut changes = half;
    if rows > src_rows {
        changes |= ROWS_ADDED;
    }
    if cols > src_cols {
        changes |= ROW_TAILS_ADDED;
    }
    let offset = g.offset + if half == FUSED_SECOND_HALF { src_bytes } else { 0 };
    let p = Piece {
        name: g.name.clone(),
        ty: g.ty,
        rows: rows as u32,
        cols: cols as u32,
        at: 0,
        bytes: matrix_bytes(g.ty, rows, cols).ok_or_else(bad)?,
        origin: Origin { file: g.file, changes, offset },
        stride,
    };
    Ok((p, src_bytes, src_cols))
}

/// A layer's experts record (its pieces placed in the pack) and, per piece, (source bytes per expert, source row
/// length).
type LayerExperts = (Experts, [(u64, u64); 3]);

/// Layer l's routed experts, if it has any.
fn experts_of(g: &Gguf, l: u32, n_used: u32) -> Result<Option<LayerExperts>> {
    let t = |s: &str| g.tensor(&format!("blk.{l}.{s}"));
    let down = t("ffn_down_exps.weight");
    let (gate, up, fused) = match (t("ffn_gate_exps.weight"), t("ffn_up_exps.weight"), t("ffn_gate_up_exps.weight")) {
        (Some(a), Some(b), None) => (a, b, false),
        (None, None, Some(f)) => (f, f, true),
        (None, None, None) => return Ok(None),
        _ => return err(format!("layer {l}: an unexpected set of expert tensors")),
    };
    let Some(down) = down else { return err(format!("layer {l}: expert gate and up without down")) };
    let (d_in, n_expert) = (gate.dims[0], gate.dims[2]);
    let w = if fused { gate.dims[1] / 2 } else { gate.dims[1] };
    if gate.n_dims != 3
        || down.n_dims != 3
        || down.dims[2] != n_expert
        || down.dims[1] != d_in
        || down.dims[0] != w
        || (!fused && (up.dims != gate.dims))
        || (fused && gate.dims[1] % 2 != 0)
    {
        return err(format!("layer {l}: the expert tensors' shapes do not agree"));
    }
    let widen_rows = !pads_hidden_itself(&g.arch) && w % 256 != 0;
    let rows = if widen_rows { up256(w) } else { w };
    let (h1, h2) = if fused { (FUSED_FIRST_HALF, FUSED_SECOND_HALF) } else { (0, 0) };
    // gate and up read the embedding: padded with it (pads_input)
    let cols_in = if pads_input(&g.arch) { up256(d_in) } else { d_in };
    let (mut pg, sg, cg) = piece_from(gate, rows, cols_in, w, d_in, h1)?;
    let (mut pu, su, cu) = piece_from(up, rows, cols_in, w, d_in, h2)?;
    let (mut pd, sd, cd) = piece_from(down, d_in, up256(w), d_in, w, 0)?;
    if fused {
        // each half's stride is the whole fused matrix's
        let full = matrix_bytes(gate.ty, gate.dims[1], d_in)
            .ok_or_else(|| Error(format!("{}: bad type or shape", gate.name)))?;
        pg.stride = full;
        pu.stride = full;
    }
    pg.at = 0;
    pu.at = page_up(pg.bytes);
    pd.at = page_up(pu.at + pu.bytes);
    let pack = Pack { bytes: page_up(pd.at + pd.bytes), gate: pg, up: pu, down: pd };
    let x = Experts {
        layer: l,
        n_expert: n_expert as u32,
        n_used,
        d_in: cols_in as u32,
        width: rows as u32,
        first: 0,
        pack,
        crcs: Vec::new(),
    };
    Ok(Some((x, [(sg, cg), (su, cu), (sd, cd)])))
}

/// Whether the architecture's model code runs a dense feed-forward as its layer's one expert behind no router (the
/// qwen forward does: moe.rs MoeLayer::load). For a layer without expert tensors (every layer of qwen35, the leading
/// dense layers of ernie4_5-moe), the converter gathers its ffn_gate, ffn_up and ffn_down into a one-expert record.
fn dense_as_expert(arch: &str) -> bool {
    arch == "qwen35" || arch == "ernie4_5-moe" || arch == "deepseek2" || arch == "lfm2moe"
}

const DENSE_SUFFIXES: [&str; 3] = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];

fn is_dense_ffn_tensor(name: &str) -> bool {
    name.strip_prefix("blk.").and_then(|s| s.split_once('.')).is_some_and(|(_, stem)| DENSE_SUFFIXES.contains(&stem))
}

/// Layer l's dense feed-forward as a one-expert record, like experts_of (None when the layer has none).
fn dense_of(g: &Gguf, l: u32) -> Result<Option<LayerExperts>> {
    let t = |s: &str| g.tensor(&format!("blk.{l}.{s}"));
    let (gate, up, down) = match (t(DENSE_SUFFIXES[0]), t(DENSE_SUFFIXES[1]), t(DENSE_SUFFIXES[2])) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        (None, None, None) => return Ok(None),
        _ => return err(format!("layer {l}: an incomplete dense feed-forward")),
    };
    let (d_in, w) = (gate.dims[0], gate.dims[1]);
    if gate.n_dims != 2
        || up.n_dims != 2
        || down.n_dims != 2
        || up.dims != gate.dims
        || down.dims[0] != w
        || down.dims[1] != d_in
    {
        return err(format!("layer {l}: the dense feed-forward's shapes do not agree"));
    }
    let rows = if w % 256 != 0 { up256(w) } else { w };
    let (mut pg, sg, cg) = piece_from(gate, rows, d_in, w, d_in, 0)?;
    let (mut pu, su, cu) = piece_from(up, rows, d_in, w, d_in, 0)?;
    let (mut pd, sd, cd) = piece_from(down, d_in, up256(w), d_in, w, 0)?;
    pg.at = 0;
    pu.at = page_up(pg.bytes);
    pd.at = page_up(pu.at + pu.bytes);
    let pack = Pack { bytes: page_up(pd.at + pd.bytes), gate: pg, up: pu, down: pd };
    let x = Experts {
        layer: l,
        n_expert: 1,
        n_used: 1,
        d_in: d_in as u32,
        width: rows as u32,
        first: 0,
        pack,
        crcs: Vec::new(),
    };
    Ok(Some((x, [(sg, cg), (su, cu), (sd, cd)])))
}

fn fetch(f: &mut File, off: u64, buf: &mut [u8]) -> Result<()> {
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(buf)?;
    Ok(())
}

fn hash_file(path: &str) -> Result<[u8; 32]> {
    let mut f = File::open(path)?;
    let mut h = blake3::Hasher::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update_rayon(&buf[..n]);
    }
    Ok(*h.finalize().as_bytes())
}

/// Converts `src` (a GGUF, or the first part of a split one) into the ToRc file `dst`. `hash_source`: also record each
/// source file's BLAKE3 (a full extra read of the GGUF).
/// A GGUF read and laid out as a ToRc file, nothing written yet: what convert_gguf writes, and what plan_gguf checks.
struct Prepared {
    g: Gguf,
    plan: write::Plan,
    /// each resident tensor's source (bytes, columns), in the plan's order
    sources: Vec<(u64, u64)>,
    /// each MoE record's pieces' sources (bytes, columns), in the plan's order
    piece_src: Vec<[(u64, u64); 3]>,
    n_tensors: usize,
    n_layers: usize,
}

/// The file a GGUF would become, laid out in memory without writing it: its directory, regions and blocks, as
/// read::Model::open would find them in the converted file (CRCs and hashes zero). For checking a model before
/// spending the time and the disk on converting it.
pub fn plan_gguf(src: &str) -> Result<crate::read::Model> {
    let p = prepare(src, false)?;
    Ok(crate::read::Model::from_plan(src, p.plan))
}

pub fn convert_gguf(src: &str, dst: &str, hash_source: bool) -> Result<Report> {
    let t0 = Instant::now();
    let Prepared { g, plan, sources, piece_src, n_tensors, n_layers } = prepare(src, hash_source)?;

    let mut srcs = g.files.iter().map(File::open).collect::<std::io::Result<Vec<_>>>()?;
    let tens: Vec<Tensor> = plan.dir.tensors.clone();
    let xs: Vec<Experts> = plan.dir.experts.clone();
    let mut tmp = Vec::new();
    let srcs_cell = std::cell::RefCell::new(&mut srcs);
    let tmp_cell = std::cell::RefCell::new(&mut tmp);
    let h = write::write(
        dst,
        plan,
        |i, at, buf| {
            let t = &tens[i];
            let (sb, sc) = sources[i];
            let m = SourceMatrix { origin: t.origin, ty: t.ty, src_bytes: sb, src_cols: sc, dst_cols: t.dims[0] };
            let mut s = srcs_cell.borrow_mut();
            source_range(&mut s[t.origin.file as usize], &m, at, buf, &mut tmp_cell.borrow_mut())
        },
        |j, e, buf| {
            let x = &xs[j];
            let mut s = srcs_cell.borrow_mut();
            for (k, pc) in x.pack.pieces().into_iter().enumerate() {
                let (sb, sc) = piece_src[j][k];
                let origin = Origin { offset: pc.origin.offset + e as u64 * pc.stride, ..pc.origin };
                let m = SourceMatrix { origin, ty: pc.ty, src_bytes: sb, src_cols: sc, dst_cols: pc.cols as u64 };
                let at = pc.at as usize;
                source_range(
                    &mut s[pc.origin.file as usize],
                    &m,
                    0,
                    &mut buf[at..at + pc.bytes as usize],
                    &mut tmp_cell.borrow_mut(),
                )?;
            }
            Ok(())
        },
    )?;
    Ok(Report {
        file_bytes: h.file_bytes,
        tensors: n_tensors,
        moe_layers: n_layers,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

/// Reads a GGUF and lays it out: everything convert_gguf does before it writes a byte.
fn prepare(src: &str, hash_source: bool) -> Result<Prepared> {
    let g = Gguf::open(src)?;
    let meta = meta::Meta::parse(&g.metadata)?;
    let n_used = meta.arch_u32("expert_used_count").unwrap_or(0);
    let layer_of = |name: &str| -> Option<u32> { name.strip_prefix("blk.")?.split('.').next()?.parse().ok() };
    // the layers whose dense FFN the model code runs as a one-expert layer (every layer of a dense qwen35; the leading
    // dense layers of ernie4_5-moe): those without expert tensors of their own. Their FFN matrices go into expert packs
    let expert_layers: std::collections::BTreeSet<u32> =
        g.tensors.iter().filter(|t| is_expert_tensor(&t.name)).filter_map(|t| layer_of(&t.name)).collect();
    let dense: std::collections::BTreeSet<u32> = if dense_as_expert(&g.arch) {
        g.tensors
            .iter()
            .filter(|t| is_dense_ffn_tensor(&t.name))
            .filter_map(|t| layer_of(&t.name))
            .filter(|l| !expert_layers.contains(l))
            .collect()
    } else {
        Default::default()
    };
    let packed_dense = |name: &str| is_dense_ffn_tensor(name) && layer_of(name).is_some_and(|l| dense.contains(&l));

    let mut tensors = Vec::new();
    let mut sources = Vec::new();
    for t in g.tensors.iter().filter(|t| !is_expert_tensor(&t.name) && !packed_dense(&t.name)) {
        let (dims, changes) = widened_dense(&g.arch, t).unwrap_or((t.dims, 0));
        let bytes = matrix_bytes(t.ty, dims[1..t.n_dims as usize].iter().product(), dims[0])
            .ok_or_else(|| Error(format!("{}: bad type or shape", t.name)))?;
        tensors.push(Tensor {
            name: t.name.clone(),
            ty: t.ty,
            n_dims: t.n_dims,
            dims,
            offset: 0,
            bytes,
            crc: 0,
            origin: Origin { file: t.file, changes, offset: t.offset },
        });
        sources.push((t.bytes, t.dims[0]));
    }
    let mut layers: Vec<u32> = g
        .tensors
        .iter()
        .filter(|t| is_expert_tensor(&t.name))
        .filter_map(|t| t.name.strip_prefix("blk.")?.split('.').next()?.parse().ok())
        .collect();
    layers.sort_unstable();
    layers.dedup();
    let mut experts = Vec::new();
    let mut piece_src = Vec::new();
    for l in layers {
        if let Some((x, s)) = experts_of(&g, l, n_used)? {
            if n_used == 0 || n_used > x.n_expert {
                return err(format!("{}.expert_used_count is missing or larger than the experts", g.arch));
            }
            experts.push(x);
            piece_src.push(s);
        }
    }
    for &l in &dense {
        if let Some((x, s)) = dense_of(&g, l)? {
            experts.push(x);
            piece_src.push(s);
        }
    }
    // the records in layer order
    let mut order: Vec<usize> = (0..experts.len()).collect();
    order.sort_by_key(|&i| experts[i].layer);
    let experts: Vec<Experts> = order.iter().map(|&i| experts[i].clone()).collect();
    let piece_src: Vec<[(u64, u64); 3]> = order.iter().map(|&i| piece_src[i]).collect();
    let mut files = Vec::new();
    for p in &g.files {
        let bytes = std::fs::metadata(p)?.len();
        let hash = if hash_source { hash_file(p)? } else { [0; 32] };
        let name = std::path::Path::new(p).file_name().map_or(p.clone(), |n| n.to_string_lossy().into_owned());
        files.push(SourceFile { bytes, hash, name });
    }
    let source = Source { files, converter: format!("torc-format {} convert-gguf", env!("CARGO_PKG_VERSION")) };
    let dir = Directory { metadata: g.metadata.clone(), source: Some(source), tensors, experts };
    let n_tensors = dir.tensors.len();
    let n_layers = dir.experts.len();
    let plan = write::plan(dir, &g.arch)?;
    Ok(Prepared { g, plan, sources, piece_src, n_tensors, n_layers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::{image, string};

    /// A two-layer MoE with a width of 704 (Gemma's case: gate, up and down all widened) and one with fused gate_up,
    /// converted, checked and compared with its source.
    fn tiny(arch: &str, fused: bool) -> (std::path::PathBuf, String, String) {
        let dir = std::env::temp_dir().join(format!("torc-conv-{arch}-{fused}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d_in, w, n) = (64u64, 704u64, 3u64);
        let q8 = |rows: u64, cols: u64, seed: u8| -> Vec<u8> {
            (0..matrix_bytes(8, rows, cols).unwrap()).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect()
        };
        let mut ts: Vec<(String, u32, Vec<u64>, Vec<u8>)> = vec![
            ("token_embd.weight".into(), 0, vec![d_in, 5], vec![1; (d_in * 5 * 4) as usize]),
            ("blk.0.attn_q.weight".into(), 8, vec![d_in, d_in], q8(d_in, d_in, 3)),
            ("blk.0.ffn_gate_inp.weight".into(), 0, vec![d_in, n], vec![2; (d_in * n * 4) as usize]),
            ("blk.0.ffn_down_exps.weight".into(), 8, vec![w, d_in, n], q8(d_in * n, w, 5)),
            ("blk.0.ffn_gate.weight".into(), 8, vec![d_in, w], q8(w, d_in, 9)),
            ("blk.0.ffn_down.weight".into(), 8, vec![w, d_in], q8(d_in, w, 11)),
        ];
        if fused {
            ts.push(("blk.0.ffn_gate_up_exps.weight".into(), 8, vec![d_in, 2 * w, n], q8(2 * w * n, d_in, 13)));
        } else {
            ts.push(("blk.0.ffn_gate_exps.weight".into(), 8, vec![d_in, w, n], q8(w * n, d_in, 13)));
            ts.push(("blk.0.ffn_up_exps.weight".into(), 8, vec![d_in, w, n], q8(w * n, d_in, 17)));
        }
        let refs: Vec<(&str, u32, Vec<u64>, Vec<u8>)> =
            ts.iter().map(|(a, b, c, d)| (a.as_str(), *b, c.clone(), d.clone())).collect();
        let img = image(
            &[
                ("general.architecture", 8, string(arch)),
                (&format!("{arch}.expert_used_count"), 4, 2u32.to_le_bytes().to_vec()),
            ],
            &refs,
        );
        let src = dir.join("m.gguf").to_str().unwrap().to_string();
        let dst = dir.join("m.torc").to_str().unwrap().to_string();
        std::fs::write(&src, img).unwrap();
        convert_gguf(&src, &dst, true).unwrap();
        (dir, src, dst)
    }

    /// Inkling 2's mainline architecture converts through the same generic expert path; its loop tensors are ordinary
    /// resident tensors, so the file round-trips like any other MoE.
    #[test]
    fn inkling2_loop_converts() {
        let (dir, src, dst) = tiny("inkling2_loop", false);
        let m = read::Model::open(&dst).unwrap();
        assert_eq!(m.header.arch, "inkling2_loop");
        assert_eq!(m.experts[0].width, 768);
        check::check(&dst).unwrap();
        check::check_against(&dst, &src).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn widened_experts_and_dense() {
        let (dir, src, dst) = tiny("gemma4", false);
        let m = read::Model::open(&dst).unwrap();
        let x = &m.experts[0];
        assert_eq!((x.width, x.pack.gate.rows, x.pack.down.cols), (768, 768, 768));
        assert_eq!(x.pack.gate.origin.changes, ROWS_ADDED);
        assert_eq!(x.pack.down.origin.changes, ROW_TAILS_ADDED);
        assert_eq!(m.tensor("blk.0.ffn_gate.weight").unwrap().dims[1], 768);
        assert_eq!(m.tensor("blk.0.ffn_down.weight").unwrap().dims[0], 768);
        assert!(m.tensor("blk.0.ffn_gate_exps.weight").is_none());
        check::check(&dst).unwrap();
        let r = check::check_against(&dst, &src).unwrap();
        assert_eq!((r.tensors, r.packs), (5, 3));
        assert!(m.source.as_ref().unwrap().files[0].hash != [0; 32]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn down_only_and_fused() {
        let (dir, src, dst) = tiny("qwen4exp", true);
        let m = read::Model::open(&dst).unwrap();
        let x = &m.experts[0];
        // qwen4exp: gate and up stay 704 rows, down's rows grow to 768
        assert_eq!((x.width, x.pack.gate.rows, x.pack.down.cols), (704, 704, 768));
        assert_eq!(x.pack.gate.origin.changes, FUSED_FIRST_HALF);
        assert_eq!(x.pack.up.origin.changes, FUSED_SECOND_HALF);
        assert_eq!(m.tensor("blk.0.ffn_gate.weight").unwrap().dims[1], 704);
        check::check(&dst).unwrap();
        check::check_against(&dst, &src).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A dense qwen35 (no expert tensors): each layer's FFN becomes a one-expert record, widened like an expert's
    /// (704 -> 768), its matrices out of the resident tensors, and the file still matches its source byte for byte.
    #[test]
    fn dense_as_one_expert() {
        let dir = std::env::temp_dir().join(format!("torc-conv-dense-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d_in, w) = (64u64, 704u64);
        let q8 = |rows: u64, cols: u64, seed: u8| -> Vec<u8> {
            (0..matrix_bytes(8, rows, cols).unwrap()).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect()
        };
        let mut ts: Vec<(String, u32, Vec<u64>, Vec<u8>)> =
            vec![("token_embd.weight".into(), 0, vec![d_in, 5], vec![1; (d_in * 5 * 4) as usize])];
        for l in 0..2u8 {
            ts.push((format!("blk.{l}.attn_qkv.weight"), 8, vec![d_in, d_in], q8(d_in, d_in, 3 + l)));
            ts.push((format!("blk.{l}.ffn_gate.weight"), 8, vec![d_in, w], q8(w, d_in, 9 + l)));
            ts.push((format!("blk.{l}.ffn_up.weight"), 8, vec![d_in, w], q8(w, d_in, 13 + l)));
            ts.push((format!("blk.{l}.ffn_down.weight"), 8, vec![w, d_in], q8(d_in, w, 17 + l)));
        }
        let refs: Vec<(&str, u32, Vec<u64>, Vec<u8>)> =
            ts.iter().map(|(a, b, c, d)| (a.as_str(), *b, c.clone(), d.clone())).collect();
        let img = image(&[("general.architecture", 8, string("qwen35"))], &refs);
        let src = dir.join("m.gguf").to_str().unwrap().to_string();
        let dst = dir.join("m.torc").to_str().unwrap().to_string();
        std::fs::write(&src, img).unwrap();
        let r = convert_gguf(&src, &dst, false).unwrap();
        assert_eq!((r.tensors, r.moe_layers), (3, 2));
        let m = read::Model::open(&dst).unwrap();
        for (l, x) in m.experts.iter().enumerate() {
            assert_eq!((x.layer, x.n_expert, x.n_used), (l as u32, 1, 1));
            assert_eq!((x.width, x.pack.gate.rows, x.pack.down.cols), (768, 768, 768));
            assert_eq!((x.pack.gate.origin.changes, x.pack.down.origin.changes), (ROWS_ADDED, ROW_TAILS_ADDED));
            assert!(
                m.blocks
                    .iter()
                    .any(|b| b.layer == l as u32 && b.kind == BlockKind::DenseFfn && b.experts == Some(l as u32))
            );
        }
        assert!(m.tensor("blk.0.ffn_gate.weight").is_none() && m.tensor("blk.1.ffn_down.weight").is_none());
        check::check(&dst).unwrap();
        let c = check::check_against(&dst, &src).unwrap();
        assert_eq!((c.tensors, c.packs), (3, 2));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Damaged files, cut short or with a byte changed: converting a GGUF and opening and checking a .torc give an
    /// error or a result, never a panic. Every third byte of the GGUF's header and tensor list is tried; the .torc is
    /// sampled across the whole file, its directory included.
    #[test]
    fn damaged_files_are_errors() {
        let (dir, src, dst) = tiny("qwen3moe", false);
        let (gguf, torc) = (std::fs::read(&src).unwrap(), std::fs::read(&dst).unwrap());
        let mut damaged: Vec<(bool, Vec<u8>)> = Vec::new();
        for (is_gguf, good) in [(true, &gguf), (false, &torc)] {
            let (span, step) = if is_gguf { (good.len().min(2048), 3) } else { (good.len(), good.len() / 1500 + 1) };
            for cut in (0..span).step_by(step * 61).chain([good.len() - 1]) {
                damaged.push((is_gguf, good[..cut].to_vec()));
            }
            for at in (0..span).step_by(step) {
                for v in [0x00, 0xff, good[at] ^ 0x80] {
                    let mut f = good.clone();
                    f[at] = v;
                    damaged.push((is_gguf, f));
                }
            }
        }
        let (bad, out) = (dir.join("bad"), dir.join("bad.torc"));
        let (bad_s, out_s) = (bad.to_str().unwrap().to_string(), out.to_str().unwrap().to_string());
        for (i, (is_gguf, f)) in damaged.iter().enumerate() {
            std::fs::write(&bad, f).unwrap();
            std::fs::remove_file(&out).ok();
            let ran = std::panic::catch_unwind(|| {
                if *is_gguf {
                    let _ = convert_gguf(&bad_s, &out_s, false);
                } else {
                    let _ = read::Model::open(&bad_s);
                    let _ = check::check(&bad_s);
                }
            });
            assert!(ran.is_ok(), "damaged file {i} ({}) panicked", if *is_gguf { "GGUF" } else { ".torc" });
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
