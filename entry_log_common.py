"""Shared config for the consolidated purchase entry log.

One officer-facing tab ("Purchase Entry Log") in the purchase sheet captures the
union of the two old bird logs. A sync projects each new row into:
  - "Pullus Purchase Tracker" (Sheet A) -> read by discrepancy_analyzer.py
  - "Daily Purchase Log"      (Sheet B) -> read by Sheet B's Monthly Scorecards

This module is the single source of truth for the entry-tab columns and how they
map onto each target, so the three scripts (setup / sync / verify) never drift.
"""
import os
from datetime import date, datetime

import gspread
from google.oauth2.service_account import Credentials

# --- spreadsheets (env-overridable; defaults are the live sheets) -----------
# Use `or default` (not getenv's default arg) so an empty env var - which is what
# GitHub Actions passes for an unset secret - still falls back to the real value.
PURCHASE_SHEET_ID = os.getenv("PURCHASE_SHEET_ID") or "1mtpxmb-0c74gTPgKepKLZpK1cn8aqNCCXjPf85oP66g"   # Sheet A
SECONDARY_SHEET_ID = os.getenv("SECONDARY_SHEET_ID") or "1lWIJbTCiNFrTYEcBsN1vRS5970qzW-OCsZCejI3HWkg"  # Sheet B

ENTRY_TAB = os.getenv("ENTRY_SHEET_NAME") or "Purchase Entry Log"
PPT_TAB = os.getenv("PURCHASE_SHEET_NAME") or "Pullus Purchase Tracker"
DPL_TAB = os.getenv("SECONDARY_SHEET_NAME") or "Daily Purchase Log"

# Header rows differ between the two existing tabs (verified on the live sheets).
PPT_HEADER_ROW = 4   # 1-indexed; data starts row 5
DPL_HEADER_ROW = 3   # 1-indexed; data starts row 4
ENTRY_HEADER_ROW = 4  # match Sheet A so the analyzer reads it unchanged

# Officer starts logging only in the entry tab from this date; the sync owns every
# target row on/after it and never touches earlier history.
CUTOVER = date(2026, 7, 1)

SERVICE_ACCOUNT_FILE = os.path.join(
    os.path.dirname(__file__), "pullus-pipeline-40a5302e034d.json"
)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# --- entry-tab schema (the full union, in display order) --------------------
# kind: "input" (officer types) or "formula" (auto, gray).
# fmt:  text | date | datetime | int | num2 | money | hrs  (drives number format)
DATE = "DATE"
ENTRY_COLUMNS = [
    # --- when & reference ---
    {"name": DATE,                          "kind": "input",   "fmt": "date"},
    {"name": "YEAR",                        "kind": "formula", "fmt": "int"},
    {"name": "MONTH",                       "kind": "formula", "fmt": "text"},
    {"name": "INVOICE NUMBER",              "kind": "input",   "fmt": "text"},
    # --- who & where ---
    {"name": "PURCHASE OFFICER NAME",       "kind": "input",   "fmt": "text"},
    {"name": "STATE",                       "kind": "input",   "fmt": "text"},
    {"name": "LOCATION",                    "kind": "input",   "fmt": "text"},
    {"name": "CLUSTER NAME",                "kind": "input",   "fmt": "text"},
    {"name": "FARMER NAME",                 "kind": "input",   "fmt": "text"},
    {"name": "FARMER ID",                   "kind": "input",   "fmt": "text"},
    {"name": "FARMER PHONE",                "kind": "input",   "fmt": "text"},
    # --- birds bought ---
    {"name": "NUMBER OF BIRDS",             "kind": "input",   "fmt": "num2"},
    {"name": "DEFORMED BIRDS",              "kind": "input",   "fmt": "num2"},
    {"name": "MINI BIRDS",                  "kind": "input",   "fmt": "num2"},
    {"name": "MINI BOUGHT AT STANDARD RATE","kind": "input",   "fmt": "text"},
    # --- pricing ---
    {"name": "BIRD UNIT PRICE (N)",         "kind": "input",   "fmt": "money"},
    {"name": "DEFORMED BIRD UNIT PRICE (N)","kind": "input",   "fmt": "money"},
    # --- product weights (chicken/gizzard + other parts, as on the old tracker) ---
    {"name": "PURCHASED CHICKEN WEIGHT",    "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED GIZZARD WEIGHT",    "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED HEAD QUANTITY",     "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED HEAD WEIGHT",       "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED LEG QUANTITY",      "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED LEG WEIGHT",        "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED LIVER QUANTITY",    "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED LIVER WEIGHT",      "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED NECK QUANTITY",     "kind": "input",   "fmt": "num2"},
    {"name": "PURCHASED NECK WEIGHT",       "kind": "input",   "fmt": "num2"},
    # --- offtake & logistics ---
    {"name": "OFFTAKE REQUESTED VIA APP",   "kind": "input",   "fmt": "text"},
    {"name": "COMPLETED ON APP",            "kind": "input",   "fmt": "text"},
    {"name": "FAILED OFFTAKE",              "kind": "input",   "fmt": "text"},
    {"name": "OFFTAKE REQUEST DATETIME",    "kind": "input",   "fmt": "datetime"},
    {"name": "PO ARRIVAL AT FARM DATETIME", "kind": "input",   "fmt": "datetime"},
    {"name": "LOGISTICS PICKUP DATETIME",   "kind": "input",   "fmt": "datetime"},
    {"name": "COLD ROOM ARRIVAL DATETIME",  "kind": "input",   "fmt": "datetime"},
    {"name": "REQUEST TO PO ARRIVAL (HRS)", "kind": "formula", "fmt": "hrs"},
    {"name": "PO ARRIVAL TO LOGISTICS (HRS)","kind": "formula","fmt": "hrs"},
    {"name": "LOGISTICS TO COLD ROOM (HRS)","kind": "formula", "fmt": "hrs"},
    {"name": "REQUEST TO FULFILMENT (HRS)", "kind": "formula", "fmt": "hrs"},
    {"name": "OFFTAKE TO LOGISTICS PICKUP (HRS)","kind": "formula","fmt": "hrs"},
    # --- notes ---
    {"name": "NOTES",                       "kind": "input",   "fmt": "text"},
]
ENTRY_HEADERS = [c["name"] for c in ENTRY_COLUMNS]

# Yes/No dropdown columns.
ENTRY_YESNO_COLS = [
    "MINI BOUGHT AT STANDARD RATE", "OFFTAKE REQUESTED VIA APP",
    "COMPLETED ON APP", "FAILED OFFTAKE",
]

# --- Sheet B ("Daily Purchase Log") header -> entry column ------------------
# Drives the projection onto Sheet B; chicken/gizzard get renamed back to B's
# names. Any B header not listed is left blank (and the sync warns).
DPL_FROM_ENTRY = {
    # Batch codes are dropped from the entry template (always empty); leave the
    # two Sheet B columns blank rather than flag them as unmapped.
    "Batch Code WC": None,
    "Batch Code GZ": None,
    "Farmer ID": "FARMER ID",
    "Invoice Number": "INVOICE NUMBER",
    "Date": "DATE",
    "Year": "YEAR",
    "Month": "MONTH",
    "State": "STATE",
    "Location": "LOCATION",
    "Purchase Officer Name": "PURCHASE OFFICER NAME",
    "Farmer Name": "FARMER NAME",
    "Farmer Phone": "FARMER PHONE",
    "Cluster Name": "CLUSTER NAME",
    "Number of Birds": "NUMBER OF BIRDS",
    "Weight of Birds (kg)": "PURCHASED CHICKEN WEIGHT",
    "Deformed Birds": "DEFORMED BIRDS",
    "Mini Birds": "MINI BIRDS",
    "Mini Bought at Standard Rate": "MINI BOUGHT AT STANDARD RATE",
    "Bird Unit Price (N)": "BIRD UNIT PRICE (N)",
    "Deformed Bird Unit Price (N)": "DEFORMED BIRD UNIT PRICE (N)",
    "Gizzard Weight (kg)": "PURCHASED GIZZARD WEIGHT",
    "Offtake Requested via App": "OFFTAKE REQUESTED VIA APP",
    "Completed on App": "COMPLETED ON APP",
    "Offtake Request DateTime": "OFFTAKE REQUEST DATETIME",
    "PO Arrival at Farm DateTime": "PO ARRIVAL AT FARM DATETIME",
    "Logistics Pickup DateTime": "LOGISTICS PICKUP DATETIME",
    "Cold Room Arrival DateTime": "COLD ROOM ARRIVAL DATETIME",
    "Request to PO Arrival (hrs)": "REQUEST TO PO ARRIVAL (HRS)",
    "PO Arrival to Logistics (hrs)": "PO ARRIVAL TO LOGISTICS (HRS)",
    "Logistics to Cold Room (hrs)": "LOGISTICS TO COLD ROOM (HRS)",
    "Request to Fulfilment (hrs)": "REQUEST TO FULFILMENT (HRS)",
    "Offtake to Logistics Pickup (hrs)": "OFFTAKE TO LOGISTICS PICKUP (HRS)",
    "Failed Offtake": "FAILED OFFTAKE",
}
# B header -> the kind of value, so the sync knows which columns to date-format.
DPL_DATE_HEADERS = {"Date"}
DPL_DATETIME_HEADERS = {
    "Offtake Request DateTime", "PO Arrival at Farm DateTime",
    "Logistics Pickup DateTime", "Cold Room Arrival DateTime",
}

# Sheets serial-date epoch.
_EPOCH = date(1899, 12, 30)


def cutover_serial():
    return (CUTOVER - _EPOCH).days


def col_letter(n):
    """1-indexed column number -> A1 letter."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def get_credentials():
    """Dual-mode google-auth credentials (file path locally, JSON string / path
    via GOOGLE_SERVICE_ACCOUNT_JSON in CI)."""
    env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if env:
        if env.strip().startswith("{"):
            import json
            return Credentials.from_service_account_info(json.loads(env), scopes=SCOPES)
        return Credentials.from_service_account_file(env, scopes=SCOPES)
    return Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)


def authenticate():
    """gspread client using the shared credentials."""
    return gspread.authorize(get_credentials())


def is_blank(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def row_date_serial(value):
    """Return a numeric serial for a target/entry DATE cell read UNFORMATTED,
    or None if blank/unparseable. Accepts serials (number) or date strings."""
    if is_blank(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # string date (e.g. "01-Jul-2026" or "2026-07-01")
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return (datetime.strptime(value.strip(), fmt).date() - _EPOCH).days
        except ValueError:
            continue
    return None
