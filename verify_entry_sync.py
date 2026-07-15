"""Prove the sync is faithful: every entry-tab field (>= cutover) round-trips into
both target tabs cell-for-cell, no column is dropped, and totals reconcile.
Read-only. Run after sync_entry_log.py.
"""
from entry_log_common import (
    PURCHASE_SHEET_ID, SECONDARY_SHEET_ID,
    ENTRY_TAB, PPT_TAB, DPL_TAB,
    ENTRY_HEADER_ROW, PPT_HEADER_ROW, DPL_HEADER_ROW,
    ENTRY_COLUMNS, ENTRY_HEADERS, ENTRY_ONLY_COLS, DPL_FROM_ENTRY,
    authenticate, cutover_serial, row_date_serial, is_blank,
)

ENTRY_INPUTS = [c["name"] for c in ENTRY_COLUMNS if c["kind"] == "input"]


def eq(a, b):
    if is_blank(a) and is_blank(b):
        return True
    try:
        return abs(float(a) - float(b)) <= 0.01
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def managed_block(ws, header_row):
    vals = ws.get_all_values(value_render_option="UNFORMATTED_VALUE")
    headers = vals[header_row - 1]
    didx = next(i for i, h in enumerate(headers) if h.strip().lower() == "date")
    co = cutover_serial()
    rows = []
    for raw in vals[header_row:]:
        ser = row_date_serial(raw[didx]) if didx < len(raw) else None
        if ser is None or ser < co:
            continue
        rows.append({h: (raw[i] if i < len(raw) else "") for i, h in enumerate(headers)})
    return headers, rows


def main():
    gc = authenticate()
    fails = []

    # 0) static coverage: no entry input column is dropped by the mapping
    ppt_ws = gc.open_by_key(PURCHASE_SHEET_ID).worksheet(PPT_TAB)
    ppt_headers, ppt_rows = managed_block(ppt_ws, PPT_HEADER_ROW)
    dpl_ws = gc.open_by_key(SECONDARY_SHEET_ID).worksheet(DPL_TAB)
    dpl_headers, dpl_rows = managed_block(dpl_ws, DPL_HEADER_ROW)

    sink = set(ppt_headers) | set(DPL_FROM_ENTRY.values()) | ENTRY_ONLY_COLS
    dropped = [c for c in ENTRY_INPUTS if c not in sink]
    if dropped:
        fails.append(f"entry input columns not mapped to any target: {dropped}")

    # entry rows (>= cutover)
    entry_ws = gc.open_by_key(PURCHASE_SHEET_ID).worksheet(ENTRY_TAB)
    evals = entry_ws.get_all_values(value_render_option="UNFORMATTED_VALUE")
    eidx = {h: i for i, h in enumerate(evals[ENTRY_HEADER_ROW - 1])}
    co = cutover_serial()
    erows = []
    for raw in evals[ENTRY_HEADER_ROW:]:
        ser = row_date_serial(raw[eidx["DATE"]]) if eidx["DATE"] < len(raw) else None
        if ser is None or ser < co:
            continue
        erows.append({h: (raw[i] if i < len(raw) else "") for h, i in eidx.items()})

    print(f"entry rows>=cutover={len(erows)}  PPT block={len(ppt_rows)}  DPL block={len(dpl_rows)}")
    if not (len(erows) == len(ppt_rows) == len(dpl_rows)):
        fails.append("row counts differ between entry tab and target blocks")

    # 1) cell-for-cell, positional (sync writes entry order)
    n = min(len(erows), len(ppt_rows), len(dpl_rows))
    for k in range(n):
        er, pr, dr = erows[k], ppt_rows[k], dpl_rows[k]
        for h in ppt_headers:
            if h in ENTRY_HEADERS and not eq(er.get(h, ""), pr.get(h, "")):
                fails.append(f"PPT row{k+1} col '{h}': entry={er.get(h)!r} vs ppt={pr.get(h)!r}")
        for h, src in DPL_FROM_ENTRY.items():
            if h in dpl_headers and not eq(er.get(src, ""), dr.get(h, "")):
                fails.append(f"DPL row{k+1} col '{h}'<-'{src}': entry={er.get(src)!r} vs dpl={dr.get(h)!r}")

    # 2) totals reconciliation
    def tot(rows, col):
        s = 0.0
        for r in rows:
            try:
                s += float(r.get(col, 0) or 0)
            except (TypeError, ValueError):
                pass
        return round(s, 2)

    checks = [
        ("birds", tot(erows, "NUMBER OF BIRDS"), tot(ppt_rows, "NUMBER OF BIRDS"), tot(dpl_rows, "Number of Birds")),
        ("chicken wt", tot(erows, "PURCHASED CHICKEN WEIGHT"), tot(ppt_rows, "PURCHASED CHICKEN WEIGHT"), tot(dpl_rows, "Weight of Birds (kg)")),
        ("gizzard wt", tot(erows, "PURCHASED GIZZARD WEIGHT"), tot(ppt_rows, "PURCHASED GIZZARD WEIGHT"), tot(dpl_rows, "Gizzard Weight (kg)")),
    ]
    print("\ntotals (entry | PPT | DPL):")
    for label, e, p, d in checks:
        ok = abs(e - p) <= 0.01 and abs(e - d) <= 0.01
        print(f"  {label:12s} {e:>12.2f} | {p:>12.2f} | {d:>12.2f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"total mismatch {label}: entry={e} ppt={p} dpl={d}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S):"))
    for f in fails[:40]:
        print("  -", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
