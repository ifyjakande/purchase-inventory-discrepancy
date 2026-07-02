#!/usr/bin/env python3
"""
Check if source data worksheets have been modified since last update.
Uses content hash of both purchase and inventory source worksheets.
"""

import json
import os
import sys
import time
import hashlib
from google.oauth2.service_account import Credentials
import gspread
from gspread.utils import fill_gaps

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"\'')
                    os.environ[key] = value

def get_credentials():
    """Get Google API credentials from environment (caller loads the .env first)."""
    service_account_info = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if not service_account_info:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")

    try:
        # Check if it's a file path or JSON content
        if service_account_info.startswith('/') or service_account_info.endswith('.json'):
            # It's a file path - for local development
            credentials = Credentials.from_service_account_file(
                service_account_info,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.metadata.readonly'
                ]
            )
            print(f"🔑 Using service account file: {service_account_info}")
        else:
            # It's JSON content - for GitHub Actions
            credentials_dict = json.loads(service_account_info)
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.metadata.readonly'
                ]
            )
            print("🔑 Using service account from environment variable")

        return credentials

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing service account JSON: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Service account file not found: {service_account_info}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating credentials: {e}")
        sys.exit(1)

def api_call_with_backoff(call, *args, **kwargs):
    """Run a gspread call, retrying 429/500/503 with exponential backoff."""
    delay = 2
    for attempt in range(4):
        try:
            return call(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if attempt == 3 or status not in (429, 500, 503):
                raise
            print(f"API error {status}, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

def get_source_data_hash(spreadsheet, worksheet_name):
    """Get content hash of a specific worksheet (single values read)."""
    try:
        # One values_get per tab; pad exactly like get_all_values() so the
        # hash input (and the stored hashes) stays identical.
        response = api_call_with_backoff(spreadsheet.values_get, f"'{worksheet_name}'")
        all_values = fill_gaps(response.get('values', [[]]))

        # Create hash of the content
        content_str = str(all_values)
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()

        print(f"📊 Source worksheet: {worksheet_name}")
        print(f"📝 Rows with data: {len([row for row in all_values if any(cell.strip() for cell in row)])}")
        print(f"🔗 Content hash: {content_hash}")

        return content_hash

    except gspread.exceptions.APIError as e:
        print(f"❌ Error reading worksheet '{worksheet_name}': {e}")
        print("Available worksheets:")
        try:
            for ws in spreadsheet.worksheets():
                print(f"  - {ws.title}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error getting source data hash for {worksheet_name}: {e}")
        sys.exit(1)

def get_combined_source_hash(credentials, purchase_sheet_id, inventory_sheet_id,
                           purchase_sheet_name, inventory_sheet_name):
    """Get combined hash of both source sheets."""
    print("🔍 Checking both source sheets for changes...")

    # One client, one open per spreadsheet, one values read per tab
    gc = gspread.authorize(credentials)
    purchase_spreadsheet = api_call_with_backoff(gc.open_by_key, purchase_sheet_id)
    inventory_spreadsheet = api_call_with_backoff(gc.open_by_key, inventory_sheet_id)

    # Get hash for purchase sheet
    purchase_hash = get_source_data_hash(purchase_spreadsheet, purchase_sheet_name)

    # Get hash for inventory sheet
    inventory_hash = get_source_data_hash(inventory_spreadsheet, inventory_sheet_name)

    # Combine both hashes
    combined_content = f"{purchase_hash}|{inventory_hash}"
    combined_hash = hashlib.md5(combined_content.encode('utf-8')).hexdigest()

    print(f"🔗 Combined hash: {combined_hash}")

    return combined_hash, purchase_hash, inventory_hash

def load_last_hash():
    """Load the last processed content hash from file."""
    hash_file = 'last_source_hash.json'
    try:
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                data = json.load(f)
                return data.get('combined_hash'), data.get('purchase_hash'), data.get('inventory_hash')
        return None, None, None
    except Exception as e:
        print(f"⚠️  Warning: Could not load last hash: {e}")
        return None, None, None

def save_hash(combined_hash, purchase_hash, inventory_hash):
    """Save the current content hashes to file."""
    hash_file = 'last_source_hash.json'
    try:
        data = {
            'combined_hash': combined_hash,
            'purchase_hash': purchase_hash,
            'inventory_hash': inventory_hash
        }
        with open(hash_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved new combined hash: {combined_hash}")
    except Exception as e:
        print(f"❌ Error saving hash: {e}")
        sys.exit(1)

def main():
    """Main function to check for changes in source data."""
    try:
        # Only run in production/CI environment
        if not os.getenv('CI'):
            print("This script should only run in CI/production")
            sys.exit(1)

        # Load .env file first
        load_env_file()

        # Get environment variables.
        # The officer's only input surface is now the consolidated entry tab (in
        # the purchase spreadsheet), so we hash that + inventory. We deliberately
        # do NOT hash 'Pullus Purchase Tracker' / 'Daily Purchase Log' anymore —
        # those are written by sync_entry_log.py and hashing them would re-trigger
        # the pipeline off its own output.
        purchase_sheet_id = os.getenv('PURCHASE_SHEET_ID')
        inventory_sheet_id = os.getenv('INVENTORY_SHEET_ID')
        # `or default` so an unset secret (empty string in CI) still falls back
        entry_sheet_name = os.getenv('ENTRY_SHEET_NAME') or 'Purchase Entry Log'
        inventory_sheet_name = os.getenv('INVENTORY_SHEET_NAME') or 'Pullus Inventory Tracker'

        if not purchase_sheet_id or not inventory_sheet_id:
            print("❌ PURCHASE_SHEET_ID and INVENTORY_SHEET_ID environment variables must be set")
            sys.exit(1)

        print(f"🔍 Checking for changes in source worksheets:")
        print(f"  📋 Entry log: '{entry_sheet_name}'")
        print(f"  📋 Inventory: '{inventory_sheet_name}'")

        # Get credentials
        credentials = get_credentials()

        # Get current source data hashes (entry tab lives in the purchase sheet)
        current_combined_hash, current_purchase_hash, current_inventory_hash = get_combined_source_hash(
            credentials, purchase_sheet_id, inventory_sheet_id,
            entry_sheet_name, inventory_sheet_name
        )

        # Load last processed hashes
        last_combined_hash, last_purchase_hash, last_inventory_hash = load_last_hash()
        print(f"📅 Last processed combined hash: {last_combined_hash or 'Never'}")

        # Compare hashes
        if current_combined_hash != last_combined_hash:
            print("✅ Source data changes detected! Update needed.")

            # Show which sheet(s) changed
            if current_purchase_hash != last_purchase_hash:
                print(f"  📋 Purchase sheet changed: {last_purchase_hash} → {current_purchase_hash}")
            if current_inventory_hash != last_inventory_hash:
                print(f"  📋 Inventory sheet changed: {last_inventory_hash} → {current_inventory_hash}")

            save_hash(current_combined_hash, current_purchase_hash, current_inventory_hash)
            print("NEEDS_UPDATE=true")
            return True
        else:
            print("⏭️  No changes in source data detected. Skipping update.")
            print("NEEDS_UPDATE=false")
            return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()