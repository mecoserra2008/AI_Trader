"""Phase 1: research / holdout split (SPEC section 5 sample design, section 7 split row, standing rule 4).

Per asset, by calendar: the boundary is ``first bar open + 0.75 * (last bar close - first bar open)``, floored to
the start of that trade day in the asset's session (so no day bar straddles it). Bars that open before the
boundary go to ``data/research/<ASSET>.parquet``; bars that open at or after it go to
``data/holdout/<ASSET>/part-NNNN.parquet``. Boundaries are written to ``data/split.json`` and to the registry line.

Holdout lock (Windows). Before the first holdout file is written, the holdout folder gets
``icacls <holdout> /deny "%USERNAME%":(OI)(CI)(R)``: every file created in it afterwards inherits the deny, so it is
unreadable from the moment it exists, and the folder cannot be listed. Measured behaviour under that ACL
(2026-11-14, NTFS, Windows 11): reading, listing, ``stat``, appending, overwriting, renaming a file into the folder
and polars' own parquet writer all fail with access denied; creating a NEW file with ``open(path, "xb")`` and
writing it works, and so does creating a subfolder. So new data (CFDs later) is written as new part files from an
in-memory parquet buffer (``write_holdout_part``), never appended. Each write is followed by a check that opening
the new file for reading raises ``PermissionError`` (the file is closed unread if that check ever failed).

Lifting the lock is the orchestrator's job at phase 9 and is not done by any code here (see ``LIFT_COMMAND``).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from vecopen import data as D
from vecopen import gaps as G
from vecopen.bars import Session, _day_start_utc, load_sessions, trade_date
from vecopen.config import load_config
from vecopen.data import TS, DedupReport, dedup_bars, panama_adjust, read_roll_calendar, research_path, validate_bars
from vecopen.ingest import Manifest, sha256_file, utc_now, write_parquet_atomic

LIFT_COMMAND = 'icacls "{holdout}" /remove:d "%USERNAME%" /T /C'


def lift_command(holdout: Path | None = None) -> str:
    """The phase-9 command (cmd.exe syntax) that removes the deny entries. Documented, never run here."""
    return LIFT_COMMAND.format(holdout=str((holdout or D.HOLDOUT_DIR).resolve()))


# ---------------------------------------------------------------- boundary

def split_boundary(first_open: datetime, last_close: datetime, session: Session, fraction: float) -> datetime:
    raw = first_open + (last_close - first_open) * fraction
    f = pl.DataFrame({"t": [raw]}, schema={"t": TS})
    out = f.select(_day_start_utc(trade_date(pl.col("t"), session), session).alias("b"))
    return out["b"][0]


# ---------------------------------------------------------------- lock and holdout writes

def lock_holdout(holdout: Path | None = None) -> None:
    """Create the folder if needed and deny read (inherited by everything created in it) for the current user."""
    holdout = holdout or D.HOLDOUT_DIR
    holdout.parent.mkdir(parents=True, exist_ok=True)
    _mkdir_locked(holdout)
    if os.name != "nt":
        os.chmod(holdout, 0o300)  # write + traverse, no read/list (POSIX equivalent; not used on this machine)
        return
    user = os.environ.get("USERNAME") or os.getlogin()
    sddl = folder_sddl(holdout)
    if ";OW)" in sddl:
        # An OWNER RIGHTS ACE (Python's mkdir(mode=0o700) and tempfile write one) replaces the owner's implicit
        # READ_CONTROL / WRITE_DAC, so a deny-read could then not be lifted without an administrator.
        raise RuntimeError(f"{holdout} has an OWNER RIGHTS entry in its ACL ({sddl}); refusing to lock it, "
                           "because the deny could not be lifted at phase 9")
    r = subprocess.run(["icacls", str(holdout), "/deny", f"{user}:(OI)(CI)(R)"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"icacls deny failed ({r.returncode}): {r.stdout} {r.stderr}")
    assert_unreadable(holdout)


def folder_sddl(path: Path) -> str:
    """The folder's DACL as SDDL (language-independent SIDs), read before the deny is applied."""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        f"(Get-Acl -LiteralPath '{path}').Sddl"], capture_output=True, text=True, timeout=60, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"could not read the ACL of {path}: {r.stderr}")
    return r.stdout.strip()


def assert_unreadable(path: Path) -> None:
    """Raise unless reading ``path`` (file: open; folder: list) raises ``PermissionError``. Never reads content."""
    try:
        if path.suffix:  # a file: try to open it for reading; closed at once, unread, if that ever works
            with open(path, "rb"):
                pass
        else:
            os.listdir(path)
    except PermissionError:
        return
    raise RuntimeError(f"{path} is readable: the holdout lock is not in effect")


def _mkdir_locked(d: Path) -> None:
    # os.makedirs(exist_ok=True) stats an existing folder, which the deny forbids: create and ignore "exists"
    try:
        os.mkdir(d)
    except FileExistsError:
        pass


def write_holdout_part(asset: str, df: pl.DataFrame, part: int, holdout: Path | None = None) -> dict[str, Any]:
    """Write ``df`` as the NEW file ``holdout/<asset>/part-<part>.parquet`` (fails if it exists) and verify it is
    unreadable. Returns size and sha256 computed from the in-memory buffer."""
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", compression_level=6, statistics=True)
    data = buf.getvalue()
    holdout = holdout or D.HOLDOUT_DIR
    d = holdout / asset
    _mkdir_locked(d)
    p = d / f"part-{part:04d}.parquet"
    with open(p, "xb") as fh:
        fh.write(data)
    assert_unreadable(p)
    import hashlib
    return {"path": p.relative_to(holdout.parent).as_posix(), "rows": df.height, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_holdout_json(asset: str, name: str, obj: Any, holdout: Path | None = None) -> str:
    holdout = holdout or D.HOLDOUT_DIR
    d = holdout / asset
    _mkdir_locked(d)
    p = d / name
    with open(p, "xb") as fh:
        fh.write(json.dumps(obj, indent=1, default=str).encode("utf-8"))
    assert_unreadable(p)
    return p.relative_to(holdout.parent).as_posix()


# ---------------------------------------------------------------- assembly

def staged_files(source: str, asset: str) -> list[Path]:
    """Monthly files (YYYY-MM) first, then daily files (YYYY-MM-DD), each in time order."""
    d = D.STAGING_DIR / source / asset
    files = sorted(d.glob("*.parquet"))
    return sorted(files, key=lambda f: (len(f.stem) != 7, f.stem))


def assemble(source: str, asset: str) -> tuple[pl.DataFrame, DedupReport]:
    parts = []
    for f in staged_files(source, asset):
        rank = 0 if len(f.stem) == 7 else 1
        parts.append(pl.read_parquet(f).with_columns(_src_rank=pl.lit(rank, pl.Int32)))
    if not parts:
        raise FileNotFoundError(f"no staged files for {source}/{asset}")
    df = pl.concat(parts, how="vertical").with_row_index("_row")
    out, rep = dedup_bars(df)
    validate_bars(out)
    return out, rep


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.isoformat()


def _runs_json(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{k: (_iso(v) if isinstance(v, datetime) else v) for k, v in r.items()} for r in df.iter_rows(named=True)]


# ---------------------------------------------------------------- build + split

def build_split(assets: list[str], *, source: str = "binance", log=print) -> dict[str, Any]:
    """Assemble, deduplicate, classify gaps, split and lock every asset in ``assets`` that is staged and not yet
    split. Crypto gap classes compare peers, so all assets of one call are classified together."""
    cfg = load_config()
    frac = float(cfg.data["split"]["research_fraction"])
    sessions = load_sessions()
    man = Manifest()
    split = json.loads(D.SPLIT_PATH.read_text(encoding="utf-8")) if D.SPLIT_PATH.exists() else {"assets": {}}
    todo = [a for a in assets if man.assets.get(a, {}).get("status") == "staged" and a not in split["assets"]]
    log(f"build-split: {len(todo)} assets to split ({', '.join(todo)})")
    if not todo:
        return {"split": []}

    # pass 1: assemble each asset once, keep only spans, runs and boundaries
    info: dict[str, dict[str, Any]] = {}
    for a in todo:
        df, rep = assemble(source, a)
        if effective_roll(a, sessions, man) == "panama":
            df, _ = panama_adjust(df, read_roll_calendar(a))
        s = sessions.for_asset(a)
        first_open = df["ts_close"][0] - timedelta(minutes=5)
        last_close = df["ts_close"][-1]
        b = split_boundary(first_open, last_close, s, frac)
        research_ts = df.filter(pl.col("ts_close") <= b)["ts_close"]
        info[a] = {"first_open": first_open, "last_close": last_close, "boundary": b, "dedup": rep,
                   "runs": G.missing_runs(df["ts_close"]), "research_runs": G.missing_runs(research_ts),
                   "research_last": research_ts[-1], "rows": df.height,
                   "flat": G.zero_volume_runs(df),
                   "research_flat": G.zero_volume_runs(df.filter(pl.col("ts_close") <= b))}
        log(f"  {a}: {df.height} bars {first_open:%Y-%m-%d %H:%M} .. {last_close:%Y-%m-%d %H:%M}, "
            f"boundary {b:%Y-%m-%d %H:%M} UTC, dups identical {rep.identical} conflicting {rep.conflicting}")
        del df

    # pass 2: gap classes. Research: peers cut at their own boundaries. Holdout: full data (sealed in holdout/).
    # Research windows: each asset's research rows only (first bar .. last research bar), so the classes can be
    # recomputed from data/research alone. Peers already split in an earlier call join the research comparison.
    research_w = {a: G.Window(i["first_open"] + timedelta(minutes=5), i["research_last"], i["research_runs"])
                  for a, i in info.items()}
    if source == "binance":
        for a in split["assets"]:
            if split["assets"][a].get("source") == "binance" and a not in research_w:
                research_w[a] = research_window(a)
    full_w = {a: G.Window(i["first_open"] + timedelta(minutes=5), i["last_close"], i["runs"]) for a, i in info.items()}
    research_cls = classify_windows(research_w, source, sessions, man)
    full_cls = classify_windows(full_w, source, sessions, man)
    rflat_w = {a: G.Window(w.first, w.last, info[a]["research_flat"] if a in info else research_window(a, flat=True).runs)
               for a, w in research_w.items()}
    research_flat = G.classify_flat(rflat_w)
    full_flat = G.classify_flat({a: G.Window(w.first, w.last, info[a]["flat"]) for a, w in full_w.items()})
    for a in todo:
        info[a]["schedule_source"] = "24/7" if source == "binance" else cfd_schedule(a, sessions, man)[0].source

    # pass 3: write research, lock holdout, write holdout parts
    lock_holdout()
    gaps_doc = json.loads(D.GAPS_PATH.read_text(encoding="utf-8")) if D.GAPS_PATH.exists() else {"assets": {}}
    done = []
    for a in todo:
        i = info[a]
        df, _ = assemble(source, a)
        if effective_roll(a, sessions, man) == "panama":
            df, applied = panama_adjust(df, read_roll_calendar(a))
            i["rolls"] = _runs_json(applied)
        b = i["boundary"]
        ts_open = pl.col("ts_close") - pl.duration(minutes=5)
        research = df.filter(ts_open < b)
        holdout = df.filter(ts_open >= b)
        D.RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        rp = research_path(a)
        write_parquet_atomic(research, rp)
        hpart = write_holdout_part(a, holdout, 0)
        hcls = G.clip_runs(full_cls[a], b + timedelta(minutes=5), None)
        hflat = G.clip_runs(full_flat[a], b + timedelta(minutes=5), None)
        sealed = write_holdout_json(a, "gaps.json", {"asset": a, "runs": _runs_json(hcls), "flat_runs": _runs_json(hflat)})
        rcls = research_cls[a]
        gaps_doc["assets"][a] = {"session": sessions.assets[a]["convention"], "runs": _runs_json(rcls),
                                 "summary": G.summarise(rcls), "schedule": i.get("schedule_source", "24/7"),
                                 "source": source, "flat_runs": _runs_json(research_flat[a]),
                                 "flat_summary": G.summarise(research_flat[a])}
        rec = {
            "session": sessions.assets[a]["convention"], "roll": effective_roll(a, sessions, man), "source": source,
            "first_ts_open": _iso(i["first_open"]), "last_ts_close": _iso(i["last_close"]),
            "boundary_utc": _iso(b), "research_fraction_target": frac,
            "research_fraction_actual": round((b - i["first_open"]) / (i["last_close"] - i["first_open"]), 6),
            "rows_total": i["rows"],
            "dedup": {"rows_in": i["dedup"].rows_in, "rows_out": i["dedup"].rows_out,
                      "identical": i["dedup"].identical, "conflicting": i["dedup"].conflicting},
            "research": {"path": rp.relative_to(D.RESEARCH_DIR.parent).as_posix(), "rows": research.height,
                         "first_ts_close": _iso(research["ts_close"][0]), "last_ts_close": _iso(research["ts_close"][-1]),
                         "bytes": rp.stat().st_size, "sha256": sha256_file(rp)},
            "holdout": {"parts": [hpart], "gaps_sealed": sealed,
                        "gap_summary": G.summarise(hcls), "flat_summary": G.summarise(hflat)},
            "rolls": i.get("rolls"), "utc": utc_now(),
        }
        split["assets"][a] = rec
        split["lift_command_phase9"] = lift_command()
        _write_json(D.SPLIT_PATH, split)
        _write_json(D.GAPS_PATH, gaps_doc)
        m = man.assets.setdefault(a, {})
        m.update({"status": "split", "holdout_bytes": hpart["bytes"], "split_utc": rec["utc"]})
        man.save()
        shutil.rmtree(D.STAGING_DIR / source / a)
        if source == "ctrader" and m.get("ctrader_name"):
            from vecopen.ingest import CTRADER_RAW, _safe
            shutil.rmtree(CTRADER_RAW / _safe(m["ctrader_name"]), ignore_errors=True)
        log(f"  {a}: research {research.height} rows -> {rp.name}; holdout {holdout.height} rows -> "
            f"{hpart['path']} (locked); staging deleted")
        done.append(a)
        del df, research, holdout
    return {"split": done, "boundaries": {a: split["assets"][a]["boundary_utc"] for a in done}}


def effective_roll(asset: str, sessions, man: Manifest) -> str:
    """``none`` or ``panama``. ``panama_if_futures_based`` needs the metadata decision recorded by cfd-download."""
    r = sessions.roll(asset)
    if r != "panama_if_futures_based":
        return r
    dec = man.assets.get(asset, {}).get("roll_decision", {}).get("roll")
    if dec not in ("none", "panama"):
        raise RuntimeError(f"{asset}: roll is panama_if_futures_based but no metadata decision is in the manifest; "
                           "run `python -m vecopen data cfd-download` (it records it from the cTrader symbol details)")
    return dec


def research_window(asset: str, *, flat: bool = False) -> G.Window:
    """Gap window of an asset from its research file: missing runs, or zero-volume runs with ``flat``."""
    from vecopen.data import load_bars
    df = load_bars(asset, columns=["ts_close", "v"])
    ts = df["ts_close"]
    return G.Window(ts[0], ts[-1], G.zero_volume_runs(df) if flat else G.missing_runs(ts))


def classify_windows(windows: dict[str, G.Window], source: str, sessions, man: Manifest) -> dict[str, pl.DataFrame]:
    if source == "binance":
        return G.classify_crypto(windows)
    out = {}
    for a, w in windows.items():
        sched, hols = cfd_schedule(a, sessions, man)
        out[a] = G.classify_cfd(w.runs, sched, hols)
    return out


def cfd_schedule(asset: str, sessions, man: Manifest) -> tuple[G.WeeklySchedule, list[G.Holiday]]:
    """The symbol's cTrader schedule and holidays if resolved (manifest ctrader map), else nominal hours."""
    from datetime import date as _date
    row = man.d.get("ctrader", {}).get("map", {}).get("map", {}).get(asset, {})
    det = row.get("details") or {}
    if det.get("schedule"):
        tz = det.get("schedule_time_zone") or "UTC"
        sched = G.WeeklySchedule.from_intervals(tz, [tuple(x) for x in det["schedule"]], "ctrader")
        hols = [G.Holiday(_date(1970, 1, 1) + timedelta(days=int(h["date_days"])), bool(h.get("recurring")),
                          h.get("start_s"), h.get("end_s"), h.get("tz") or tz) for h in det.get("holidays", [])]
        return sched, hols
    conv = sessions.raw["conventions"][sessions.assets[asset]["convention"]]
    return G.WeeklySchedule.from_hours(conv["tz"], conv["hours"], "sessions.yaml nominal hours"), []


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
