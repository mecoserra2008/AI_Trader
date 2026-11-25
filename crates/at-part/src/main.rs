//! `part-book <ob200.zip | lines.jsonl> <out.f64> [--bin-ms 1000] [--near-bp 5]`
//!
//! Writes one row of `at_part::N_COLS` little-endian f64 per emitted bin (column order = BOOK_COLS of
//! ai_trader.research.participants) and prints the QA counters as JSON on stdout.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, bail};
use flate2::read::DeflateDecoder;

fn open_lines(path: &Path) -> anyhow::Result<Box<dyn BufRead>> {
    let f = File::open(path).with_context(|| path.display().to_string())?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        let mut h = [0u8; 30];
        r.read_exact(&mut h)?;
        anyhow::ensure!(&h[0..4] == b"PK\x03\x04", "{}: not a zip local file header", path.display());
        let method = u16::from_le_bytes([h[8], h[9]]);
        let csize = u32::from_le_bytes([h[18], h[19], h[20], h[21]]) as u64;
        let skip = u16::from_le_bytes([h[26], h[27]]) as usize + u16::from_le_bytes([h[28], h[29]]) as usize;
        let mut buf = vec![0u8; skip];
        r.read_exact(&mut buf)?;
        return match method {
            8 => Ok(Box::new(BufReader::with_capacity(1 << 20, DeflateDecoder::new(r)))),
            0 if csize != 0xFFFF_FFFF && csize > 0 => Ok(Box::new(BufReader::new(r.take(csize)))),
            m => bail!("{}: unsupported zip member (method {m})", path.display()),
        };
    }
    Ok(Box::new(r))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: part-book <ob200.zip> <out.f64> [--bin-ms N] [--near-bp X]");
    }
    let mut bin_ms = 1000i64;
    let mut near_bp = 5.0f64;
    let mut i = 3;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--bin-ms" => bin_ms = args[i + 1].parse()?,
            "--near-bp" => near_bp = args[i + 1].parse()?,
            o => bail!("unknown option {o}"),
        }
        i += 2;
    }
    let rd = open_lines(Path::new(&args[1]))?;
    let tmp = format!("{}.part", args[2]);
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    let mut rows = 0u64;
    let mut err: Option<std::io::Error> = None;
    let qa = at_part::reduce_lines(rd, bin_ms, near_bp, |r| {
        if err.is_none() {
            for v in r {
                if let Err(e) = out.write_all(&v.to_le_bytes()) {
                    err = Some(e);
                    break;
                }
            }
            rows += 1;
        }
    })?;
    if let Some(e) = err {
        return Err(e.into());
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp, &args[2])?;
    println!(
        "{{\"messages\":{},\"snapshots\":{},\"before_snapshot\":{},\"u_gaps\":{},\"bad_lines\":{},\"rows\":{}}}",
        qa.messages, qa.snapshots, qa.before_snapshot, qa.u_gaps, qa.bad_lines, rows
    );
    Ok(())
}
