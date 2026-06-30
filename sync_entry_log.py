"""Project new rows from the consolidated "Purchase Entry Log" into the two
existing tabs so the current pipeline runs unchanged:

  Purchase Entry Log (Sheet A)
        |--> Pullus Purchase Tracker (Sheet A)  -> discrepancy_analyzer.py
        '--> Daily Purchase Log      (Sheet B)  -> Monthly Scorecards

The sync owns every target row on/after CUTOVER and rewrites that block from the
entry tab each run. Pre-cutover history is never touched. Idempotent.
"""
from entry_log_common import (
    PURCHASE_SHEET_ID, SECONDARY_SHEET_ID,
    ENTRY_TAB, PPT_TAB, DPL_TAB,
    ENTRY_HEADER_ROW, PPT_HEADER_ROW, DPL_HEADER_ROW,
    ENTRY_HEADERS, DPL_FROM_ENTRY, DPL_DATE_HEADERS, DPL_DATETIME_HEADERS,
    authenticate, col_letter, cutover_serial, row_date_serial,
)


def read_entry_rows(gc):
    """Return list of dicts {entry_header: native_value} for entry rows with a
    DATE on/after cutover. Read UNFORMATTED so numbers/serials stay native."""
    ws = gc.open_by_key(PURCHASE_SHEET_ID).worksheet(ENTRY_TAB)
    vals = ws.get_all_values(value_render_option="UNFORMATTED_VALUE")
    headers = vals[ENTRY_HEADER_ROW - 1] if len(vals) >= ENTRY_HEADER_ROW else []
    idx = {h: i for i, h in enumerate(headers)}
    date_i = idx.get("DATE")
    co = cutover_serial()
    rows = []
    for raw in vals[ENTRY_HEADER_ROW:]:
        if date_i is None or date_i >= len(raw):
            continue
        ser = row_date_serial(raw[date_i])
        if ser is None or ser < co:
            continue
        rows.append({h: (raw[i] if i < len(raw) else "") for h, i in idx.items()})
    return rows


def _project(entry_rows, target_headers, mapping=None):
    """mapping: target_header -> entry_header (None => identity by name)."""
    out = []
    for er in entry_rows:
        row = []
        for h in target_headers:
            src = mapping.get(h) if mapping else h
            row.append(er.get(src, "") if src else "")
        out.append(row)
    return out


def _pad(row, n):
    row = list(row[:n])
    return row + [""] * (n - len(row))


def _sync_target(ws, header_row, target_headers, projected, date_cols, datetime_cols):
    """Reconcile the cutover-onward (>= CUTOVER) block of a target tab from the
    entry tab, WITHOUT ever touching earlier-month history.

    Safety model:
      - Rows dated before CUTOVER are never read for content nor overwritten...
      - ...except: if an earlier-month row was added *below* the synced block
        (e.g. a late June entry after July rows already landed), it is preserved
        and floated to the top of the block - never lost, never duplicated.
      - The block is always rewritten as one contiguous, gap-free unit, so the
        analyzer never sees blank rows.
    """
    n_cols = len(target_headers)
    last_col = col_letter(n_cols)
    vals = ws.get_all_values(value_render_option="UNFORMATTED_VALUE")

    date_idx = next(i for i, h in enumerate(target_headers) if h.strip().lower() == "date")
    co = cutover_serial()

    managed_pos = []
    last_pre = header_row        # last row dated before cutover (real history)
    last_content = header_row    # last row with any content at all
    info = {}
    for rr in range(header_row + 1, len(vals) + 1):       # 1-indexed sheet rows
        raw = vals[rr - 1]
        has_content = any(str(c).strip() != "" for c in raw)
        ser = row_date_serial(raw[date_idx]) if date_idx < len(raw) else None
        info[rr] = (ser, raw, has_content)
        if has_content:
            last_content = rr
            if ser is None or ser < co:
                last_pre = rr
        if ser is not None and ser >= co:
            managed_pos.append(rr)

    # block starts right after the real history, or at the first synced row if one
    # already sits higher (handles late earlier-month rows added below the block)
    tail_start = last_pre + 1
    if managed_pos:
        tail_start = min(tail_start, min(managed_pos))

    # any pre-cutover row that ended up inside the block -> keep it, float it up
    preserved = [_pad(info[rr][1], n_cols) for rr in range(tail_start, last_content + 1)
                 if info.get(rr, (None, [], False))[2]
                 and (info[rr][0] is None or info[rr][0] < co)]
    if preserved:
        print(f"  note: {ws.title}: kept {len(preserved)} earlier-month row(s) found "
              f"below the block (floated above it; nothing lost/duplicated).")

    new_block = preserved + projected
    if new_block:
        end = tail_start + len(new_block) - 1
        ws.update(range_name=f"A{tail_start}:{last_col}{end}",
                  values=new_block, value_input_option="RAW")
        reqs = []
        for h in date_cols:
            reqs.append(_fmt_req(ws.id, tail_start - 1, end, target_headers.index(h),
                                 "DATE", "dd-mmm-yyyy"))
        for h in datetime_cols:
            reqs.append(_fmt_req(ws.id, tail_start - 1, end, target_headers.index(h),
                                 "DATE_TIME", "dd-mmm-yyyy hh:mm:ss am/pm"))
        if reqs:
            ws.spreadsheet.batch_update({"requests": reqs})
    else:
        end = tail_start - 1

    # clear any leftover rows below the rewritten block (shrunk block / removed rows)
    if last_content > end:
        ws.batch_clear([f"A{end + 1}:{last_col}{last_content}"])

    return len(projected), tail_start, end


def _fmt_req(sid, start_row0, end_row1, col0, ftype, pattern):
    return {"repeatCell": {
        "range": {"sheetId": sid, "startRowIndex": start_row0, "endRowIndex": end_row1,
                  "startColumnIndex": col0, "endColumnIndex": col0 + 1},
        "cell": {"userEnteredFormat": {"numberFormat": {"type": ftype, "pattern": pattern}}},
        "fields": "userEnteredFormat.numberFormat"}}


def main():
    gc = authenticate()
    entry_rows = read_entry_rows(gc)
    print(f"Entry tab: {len(entry_rows)} row(s) on/after cutover.")

    # --- Sheet A: Pullus Purchase Tracker (identity mapping by name) ---
    sh_a = gc.open_by_key(PURCHASE_SHEET_ID)
    ppt = sh_a.worksheet(PPT_TAB)
    ppt_headers = ppt.get_all_values()[PPT_HEADER_ROW - 1]
    missing = [h for h in ppt_headers if h not in ENTRY_HEADERS and h.strip()]
    if missing:
        print(f"  note: PPT columns not in entry schema (left blank): {missing}")
    ppt_proj = _project(entry_rows, ppt_headers)
    n, s, e = _sync_target(ppt, PPT_HEADER_ROW, ppt_headers, ppt_proj,
                           date_cols=["DATE"], datetime_cols=[])
    print(f"  '{PPT_TAB}': wrote {n} rows (rows {s}-{e}).")

    # --- Sheet B: Daily Purchase Log (renamed mapping) ---
    sh_b = gc.open_by_key(SECONDARY_SHEET_ID)
    dpl = sh_b.worksheet(DPL_TAB)
    dpl_headers = dpl.get_all_values()[DPL_HEADER_ROW - 1]
    unmapped = [h for h in dpl_headers if h.strip() and h not in DPL_FROM_ENTRY]
    if unmapped:
        print(f"  note: Daily Purchase Log columns with no entry source (left blank): {unmapped}")
    dpl_proj = _project(entry_rows, dpl_headers, DPL_FROM_ENTRY)
    date_cols = [h for h in dpl_headers if h in DPL_DATE_HEADERS]
    dt_cols = [h for h in dpl_headers if h in DPL_DATETIME_HEADERS]
    n, s, e = _sync_target(dpl, DPL_HEADER_ROW, dpl_headers, dpl_proj,
                           date_cols=date_cols, datetime_cols=dt_cols)
    print(f"  '{DPL_TAB}': wrote {n} rows (rows {s}-{e}).")
    print("Sync complete.")


if __name__ == "__main__":
    main()
