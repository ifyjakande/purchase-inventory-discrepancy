"""Create the consolidated "Purchase Entry Log" tab in the purchase sheet, and
(with --widen) grow Sheet B so synced rows keep feeding the Monthly Scorecards.

Usage:
    python setup_entry_tab.py            # create / rebuild the entry tab
    python setup_entry_tab.py --widen    # also widen scorecard ranges + grow DPL
    python setup_entry_tab.py --force    # rebuild even if the tab holds data

WARNING: a rebuild CLEARS the entry tab, including any officer data logged since
cutover. If the tab has data rows the script refuses to run without --force;
back the rows up first (they are also mirrored in the two target tabs by the
sync, but only rows >= CUTOVER). It never touches the data tabs themselves.
"""
import re
import sys

import gspread
from googleapiclient.discovery import build

from entry_log_common import (
    PURCHASE_SHEET_ID, SECONDARY_SHEET_ID, ENTRY_TAB, PPT_TAB, DPL_TAB,
    PPT_HEADER_ROW, DPL_HEADER_ROW,
    ENTRY_COLUMNS, ENTRY_HEADERS, ENTRY_HEADER_ROW, ENTRY_YESNO_COLS,
    authenticate, get_credentials, col_letter,
)

LAST_DATA_ROW = ENTRY_HEADER_ROW + 2000   # pre-fill formulas through here
SCORECARD_NEW_BOUND = 5000                 # widen 'Daily Purchase Log' ranges to this

# columns that get a suggestion dropdown built from the live data
# (CLUSTER NAME and LOCATION are intentionally free text - too many/messy values)
CATEGORICAL_COLS = ["PURCHASE OFFICER NAME", "STATE"]

NUMBER_FORMATS = {
    "text": None,  # handled as TEXT type
    "date": ("DATE", "dd-mmm-yyyy"),
    "datetime": ("DATE_TIME", "dd-mmm-yyyy hh:mm:ss am/pm"),
    "int": ("NUMBER", "0"),
    "num2": ("NUMBER", "0.00"),
    "money": ("NUMBER", "#,##0.00"),
    "hrs": ("NUMBER", "0.00"),
}


def name_to_letter():
    return {c["name"]: col_letter(i + 1) for i, c in enumerate(ENTRY_COLUMNS)}


def build_formula(name, letters, r):
    L = letters
    d = L["DATE"]
    off = L["OFFTAKE REQUEST DATETIME"]
    po = L["PO ARRIVAL AT FARM DATETIME"]
    lo = L["LOGISTICS PICKUP DATETIME"]
    cr = L["COLD ROOM ARRIVAL DATETIME"]
    if name == "YEAR":
        return f'=IF(${d}{r}="","",YEAR(${d}{r}))'
    if name == "MONTH":
        return f'=IF(${d}{r}="","",TEXT(${d}{r},"mmmm"))'
    pairs = {
        "REQUEST TO PO ARRIVAL (HRS)": (off, po),
        "PO ARRIVAL TO LOGISTICS (HRS)": (po, lo),
        "LOGISTICS TO COLD ROOM (HRS)": (lo, cr),
        "REQUEST TO FULFILMENT (HRS)": (off, cr),
        "OFFTAKE TO LOGISTICS PICKUP (HRS)": (off, lo),
    }
    a, b = pairs[name]
    return f'=IF(OR(${a}{r}="",${b}{r}=""),"",(${b}{r}-${a}{r})*24)'


def _source_validation_lists(svc, sid, tab, header_row, n_cols, n_data=20):
    """Return {header: [values]} for every ONE_OF_LIST validation on a tab, so we
    inherit the canonical dropdown options defined on the source sheets (which may
    include valid values not yet present in the data)."""
    rng = f"'{tab}'!A{header_row}:{col_letter(n_cols)}{header_row + n_data}"
    try:
        r = svc.spreadsheets().get(spreadsheetId=sid, ranges=[rng], includeGridData=True,
            fields="sheets(data(rowData(values(dataValidation,userEnteredValue))))").execute()
    except Exception:
        return {}
    rows = r["sheets"][0]["data"][0].get("rowData", [])
    if not rows:
        return {}
    headers = [c.get("userEnteredValue", {}).get("stringValue", "") for c in rows[0].get("values", [])]
    out = {}
    for row in rows[1:]:
        for ci, cell in enumerate(row.get("values", [])):
            dv = cell.get("dataValidation") or {}
            if dv.get("condition", {}).get("type") == "ONE_OF_LIST":
                h = headers[ci] if ci < len(headers) else str(ci)
                out.setdefault(h, [v.get("userEnteredValue") for v in dv["condition"]["values"]])
    return out


def build_dropdown_lists(gc):
    """Complete officer / state / cluster / location dropdown lists = union of the
    canonical validation lists defined on the source sheets + the distinct values
    actually present in the data. Deduped case-insensitively, sorted."""
    ppt = gc.open_by_key(PURCHASE_SHEET_ID).worksheet(PPT_TAB).get_all_values()
    dpl = gc.open_by_key(SECONDARY_SHEET_ID).worksheet(DPL_TAB).get_all_values()
    ph, dh = ppt[PPT_HEADER_ROW - 1], dpl[DPL_HEADER_ROW - 1]

    def col(rows, hdr, name, start):
        if name not in hdr:
            return []
        i = hdr.index(name)
        return [r[i].strip() for r in rows[start:] if len(r) > i and r[i].strip()]

    def dedup(values):
        counts, canon = {}, {}
        for v in values:
            if not v or not str(v).strip():
                continue
            v = str(v).strip()
            k = v.lower()
            counts[k] = counts.get(k, 0) + 1
            if k not in canon or counts[k] > counts.get(canon[k].lower(), 0):
                canon[k] = v
        return sorted(canon.values(), key=str.lower)

    pds, dds = PPT_HEADER_ROW, DPL_HEADER_ROW
    # canonical validation lists defined on every input tab across both sheets
    svc = build("sheets", "v4", credentials=get_credentials(), cache_discovery=False)
    vP = _source_validation_lists(svc, PURCHASE_SHEET_ID, PPT_TAB, PPT_HEADER_ROW, 20)
    vD = _source_validation_lists(svc, SECONDARY_SHEET_ID, DPL_TAB, DPL_HEADER_ROW, 33)
    vE = _source_validation_lists(svc, SECONDARY_SHEET_ID, "Daily Egg Purchase Log", 3, 18)
    vC = _source_validation_lists(svc, SECONDARY_SHEET_ID, "Competitor Price Log", 3, 10)

    # STATE: every state-field validation list (incl. PPT's mislabeled LOCATION
    # list, which actually holds states) + data
    state_src = (vP.get("STATE", []) + vP.get("LOCATION", []) + vD.get("State", [])
                 + vE.get("State", []) + vC.get("State", []))
    # OFFICER: every officer validation list + data
    officer_src = (vP.get("PURCHASE OFFICER NAME", []) + vD.get("Purchase Officer Name", [])
                   + vE.get("Purchase Officer Name", []))
    # CLUSTER: drop the "Cluster A/B/C/D" placeholders from Sheet B; use the real
    # cluster names from the data, plus "Unclustered"
    cluster_src = col(dpl, dh, "Cluster Name", dds) + ["Unclustered"]
    # LOCATION: Sheet A's LOCATION dropdown is a (wrong) states list, so ignore it
    # and use the real location names from the data
    location_src = col(dpl, dh, "Location", dds) + col(ppt, ph, "LOCATION", pds)

    return {
        "PURCHASE OFFICER NAME": dedup(officer_src
                                       + col(ppt, ph, "PURCHASE OFFICER NAME", pds)
                                       + col(dpl, dh, "Purchase Officer Name", dds)),
        "STATE": dedup(state_src + col(ppt, ph, "STATE", pds) + col(dpl, dh, "State", dds)),
        "CLUSTER NAME": dedup(cluster_src),
        "LOCATION": dedup(location_src),
    }


def width_for(c):
    name, fmt = c["name"], c["fmt"]
    if name == "NOTES":
        return 240
    if fmt == "datetime":
        return 165
    if fmt == "date":
        return 105
    if name in ("PURCHASE OFFICER NAME", "FARMER NAME", "LOCATION", "CLUSTER NAME"):
        return 150
    if name in ENTRY_YESNO_COLS:
        return 95
    if fmt in ("int", "num2", "hrs", "money"):
        return 115
    return 125


def get_or_create(sh, title, rows, cols):
    try:
        ws = sh.worksheet(title)
        ws.clear()
        # drop any leftover formatting/validation from a previous build
        sh.batch_update({"requests": [
            {"updateCells": {"range": {"sheetId": ws.id}, "fields": "userEnteredFormat,dataValidation"}}
        ]})
        ws.resize(rows=rows, cols=cols)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
    return ws


def create_entry_tab(gc, force=False):
    sh = gc.open_by_key(PURCHASE_SHEET_ID)
    n_cols = len(ENTRY_HEADERS)

    # A rebuild clears the tab, so refuse to destroy live officer data.
    try:
        existing = sh.worksheet(ENTRY_TAB)
        vals = existing.get_all_values()
        n_data = sum(1 for r in vals[ENTRY_HEADER_ROW:] if any(c.strip() for c in r))
        if n_data and not force:
            raise SystemExit(
                f"ABORTED: '{ENTRY_TAB}' holds {n_data} data row(s); rebuilding would "
                f"erase them. Back them up, then re-run with --force.")
    except gspread.WorksheetNotFound:
        pass

    ws = get_or_create(sh, ENTRY_TAB, LAST_DATA_ROW, n_cols)
    last_col = col_letter(n_cols)
    letters = name_to_letter()

    title = "PULLUS PURCHASE - Consolidated Daily Entry Log"
    legend = ("[White] = you type   |   [Gray] = auto-calculated (do not edit).   "
              "Log every bird purchase here once. YEAR, MONTH and all (HRS) fill in automatically.")

    header_block = [
        [title] + [""] * (n_cols - 1),
        [legend] + [""] * (n_cols - 1),
        [""] * n_cols,
        ENTRY_HEADERS,
    ]
    ws.update(range_name=f"A1:{last_col}{ENTRY_HEADER_ROW}", values=header_block)

    # Pre-fill the formula columns down to LAST_DATA_ROW (USER_ENTERED).
    value_ranges = []
    for i, c in enumerate(ENTRY_COLUMNS):
        if c["kind"] != "formula":
            continue
        L = col_letter(i + 1)
        col_vals = [[build_formula(c["name"], letters, r)]
                    for r in range(ENTRY_HEADER_ROW + 1, LAST_DATA_ROW + 1)]
        value_ranges.append({"range": f"{L}{ENTRY_HEADER_ROW + 1}:{L}{LAST_DATA_ROW}",
                             "values": col_vals})
    ws.batch_update(value_ranges, value_input_option="USER_ENTERED")

    dropdowns = build_dropdown_lists(gc)
    _format_entry_tab(sh, ws, n_cols, dropdowns)
    return ws


def _format_entry_tab(sh, ws, n_cols, dropdowns):
    sid = ws.id
    reqs = []

    # merge + style title / legend
    for r in (0, 1):
        reqs.append({"mergeCells": {"range": {"sheetId": sid, "startRowIndex": r, "endRowIndex": r + 1,
                     "startColumnIndex": 0, "endColumnIndex": n_cols}, "mergeType": "MERGE_ALL"}})
    reqs.append({"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1,
                "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": {"red": 0.12, "green": 0.29, "blue": 0.49},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE",
                "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "fontSize": 14, "bold": True}}},
                "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment,textFormat)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 1, "endRowIndex": 2,
                "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": {"red": 0.93, "green": 0.95, "blue": 0.98},
                "horizontalAlignment": "CENTER", "wrapStrategy": "WRAP",
                "textFormat": {"foregroundColor": {"red": 0.25, "green": 0.25, "blue": 0.25}, "fontSize": 10, "italic": True}}},
                "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,wrapStrategy,textFormat)"}})

    # header row (row index 3)
    reqs.append({"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 3, "endRowIndex": 4,
                "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": {"red": 0.22, "green": 0.46, "blue": 0.72},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE", "wrapStrategy": "WRAP",
                "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "fontSize": 10, "bold": True}}},
                "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment,wrapStrategy,textFormat)"}})

    # freeze through header
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": sid,
                "gridProperties": {"frozenRowCount": 4}}, "fields": "gridProperties.frozenRowCount"}})

    # per-column number formats + gray shading on auto (formula) columns
    for i, c in enumerate(ENTRY_COLUMNS):
        rng = {"sheetId": sid, "startRowIndex": ENTRY_HEADER_ROW, "endRowIndex": LAST_DATA_ROW,
               "startColumnIndex": i, "endColumnIndex": i + 1}
        fmt = {}
        nf = NUMBER_FORMATS[c["fmt"]]
        if c["fmt"] == "text":
            fmt["numberFormat"] = {"type": "TEXT"}
        elif nf:
            fmt["numberFormat"] = {"type": nf[0], "pattern": nf[1]}
        if c["kind"] == "formula":
            fmt["backgroundColor"] = {"red": 0.91, "green": 0.91, "blue": 0.91}
        fields = []
        if "numberFormat" in fmt:
            fields.append("numberFormat")
        if "backgroundColor" in fmt:
            fields.append("backgroundColor")
        if fields:
            reqs.append({"repeatCell": {"range": rng, "cell": {"userEnteredFormat": fmt},
                        "fields": "userEnteredFormat(" + ",".join(fields) + ")"}})

    name_idx = {c["name"]: i for i, c in enumerate(ENTRY_COLUMNS)}

    def validation(i, values, strict):
        reqs.append({"setDataValidation": {"range": {"sheetId": sid, "startRowIndex": ENTRY_HEADER_ROW,
                    "endRowIndex": LAST_DATA_ROW, "startColumnIndex": i, "endColumnIndex": i + 1},
                    "rule": {"condition": {"type": "ONE_OF_LIST",
                    "values": [{"userEnteredValue": v} for v in values]},
                    "showCustomUi": True, "strict": strict}}})

    # Yes/No dropdowns (strict - only Yes/No allowed)
    for col in ENTRY_YESNO_COLS:
        validation(name_idx[col], ["Yes", "No"], strict=True)

    # categorical suggestion dropdowns (officer / state / cluster / location);
    # non-strict so a new name can still be typed, but the list is one click away
    for col in CATEGORICAL_COLS:
        vals = dropdowns.get(col, [])
        if vals:
            validation(name_idx[col], vals, strict=False)

    # explicit per-column widths (no autoresize - keeps numbers tight, names wide)
    for i, c in enumerate(ENTRY_COLUMNS):
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "COLUMNS",
                    "startIndex": i, "endIndex": i + 1},
                    "properties": {"pixelSize": width_for(c)}, "fields": "pixelSize"}})

    # row heights: tall title, two-line header
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS",
                "startIndex": 0, "endIndex": 1}, "properties": {"pixelSize": 42}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS",
                "startIndex": 1, "endIndex": 2}, "properties": {"pixelSize": 32}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS",
                "startIndex": 3, "endIndex": 4}, "properties": {"pixelSize": 54}, "fields": "pixelSize"}})

    sh.batch_update({"requests": reqs})


def widen_scorecards(gc):
    """Grow 'Daily Purchase Log' capacity and extend the Monthly Scorecards
    SUMIFS/AVERAGEIFS ranges that read it, so synced rows aren't dropped."""
    sh = gc.open_by_key(SECONDARY_SHEET_ID)

    dpl = sh.worksheet(DPL_TAB)
    if dpl.row_count < SCORECARD_NEW_BOUND:
        dpl.resize(rows=SCORECARD_NEW_BOUND)
        print(f"  grew '{DPL_TAB}' to {SCORECARD_NEW_BOUND} rows")

    sc = sh.worksheet("Monthly Scorecards")
    formulas = sc.get_all_values(value_render_option="FORMULA")
    pat = re.compile(r"('Daily Purchase Log'![^,)]*?\$)2003\b")
    updates = []
    changed = 0
    for r, row in enumerate(formulas, start=1):
        for cidx, val in enumerate(row):
            if isinstance(val, str) and "'Daily Purchase Log'!" in val and "2003" in val:
                new = pat.sub(lambda m: m.group(1) + str(SCORECARD_NEW_BOUND), val)
                if new != val:
                    updates.append({"range": f"{col_letter(cidx + 1)}{r}", "values": [[new]]})
                    changed += 1
    if updates:
        sc.batch_update(updates, value_input_option="USER_ENTERED")
    print(f"  rewrote {changed} Monthly Scorecards formula cells (2003 -> {SCORECARD_NEW_BOUND})")


def main():
    gc = authenticate()
    print(f"Creating entry tab '{ENTRY_TAB}'...")
    ws = create_entry_tab(gc, force="--force" in sys.argv)
    print(f"  done: https://docs.google.com/spreadsheets/d/{PURCHASE_SHEET_ID}/edit#gid={ws.id}")
    if "--widen" in sys.argv:
        print("Widening Monthly Scorecards ranges...")
        widen_scorecards(gc)
    print("Setup complete.")


if __name__ == "__main__":
    main()
