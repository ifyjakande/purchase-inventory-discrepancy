import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import os
import json
import time
import random
import pytz
from datetime import datetime

def load_env_file():
    """Load environment variables from .env file if it exists (for local development)."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"\'')
                    # Only set if not already in environment (production takes precedence)
                    if key not in os.environ:
                        os.environ[key] = value

# Products tracked across the purchase and inventory sheets, in display order.
# 'label' drives the report column names (e.g. "Purchase Chicken Weight"),
# so chicken/gizzard keep their exact existing column names.
# To add a product later, just add an entry here - all reports pick it up.
PRODUCTS = [
    {'label': 'Chicken',    'purchase_col': 'PURCHASED CHICKEN WEIGHT',     'inventory_col': 'INVENTORY CHICKEN WEIGHT'},
    {'label': 'Gizzard',    'purchase_col': 'PURCHASED GIZZARD WEIGHT',     'inventory_col': 'INVENTORY GIZZARD WEIGHT'},
    {'label': 'Head',       'purchase_col': 'PURCHASED HEAD WEIGHT',        'inventory_col': 'INVENTORY HEAD WEIGHT'},
    {'label': 'Leg',        'purchase_col': 'PURCHASED LEG WEIGHT',         'inventory_col': 'INVENTORY LEG WEIGHT'},
    {'label': 'Liver',      'purchase_col': 'PURCHASED LIVER WEIGHT',       'inventory_col': 'INVENTORY LIVER WEIGHT'},
    {'label': 'Neck',       'purchase_col': 'PURCHASED NECK WEIGHT',        'inventory_col': 'INVENTORY NECK WEIGHT'},
]

class DiscrepancyAnalyzer:
    def __init__(self, service_account_json=None):
        """Initialize with service account credentials"""
        self.gc = self._authenticate(service_account_json)
        self.purchase_targets_2026 = self._load_purchase_targets()
        # Products shown in the reports. Defaults to all; run_analysis narrows this
        # to only the products that actually have data so empty columns don't clutter.
        self.active_products = list(PRODUCTS)

    def _compute_active_products(self, purchase_df, inventory_df):
        """Return only the products that have any data on either sheet, so the
        reports don't show empty columns for products no one has recorded yet."""
        active = []
        for p in PRODUCTS:
            p_sum = float(purchase_df[p['purchase_col']].abs().sum()) if p['purchase_col'] in purchase_df.columns else 0
            i_sum = float(inventory_df[p['inventory_col']].abs().sum()) if p['inventory_col'] in inventory_df.columns else 0
            if p_sum > 0 or i_sum > 0:
                active.append(p)
        return active

    def _get_wat_timestamp(self):
        """Get current timestamp in West Africa Time with AM/PM format"""
        wat_tz = pytz.timezone('Africa/Lagos')
        now = datetime.now(wat_tz)
        return now.strftime('%b %d, %Y at %I:%M %p WAT')
    
    def _api_call_with_retry(self, func, max_retries=3, base_delay=1):
        """Execute API call with exponential backoff retry for rate limiting and server errors"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                # Extract status code from different error types
                status_code = None
                error_message = str(e).lower()
                
                # Check HttpError from googleapiclient
                if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
                    status_code = int(e.resp.status)
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                elif 'quota exceeded' in error_message or 'rate limit' in error_message or '429' in error_message:
                    status_code = 429
                elif 'internal error' in error_message or '500' in error_message:
                    status_code = 500
                elif '502' in error_message or 'bad gateway' in error_message:
                    status_code = 502
                elif '503' in error_message or 'service unavailable' in error_message:
                    status_code = 503
                elif '504' in error_message or 'gateway timeout' in error_message:
                    status_code = 504
                
                # Determine retry strategy based on error type
                should_retry = False
                retry_strategy = None
                
                if status_code == 429:
                    # Rate limiting - use full retry count with exponential backoff
                    should_retry = attempt < max_retries - 1
                    retry_strategy = "rate_limit"
                elif status_code in [500, 502, 503, 504]:
                    # Server errors - use limited retries (max 2) with shorter backoff
                    server_max_retries = min(2, max_retries)
                    should_retry = attempt < server_max_retries - 1
                    retry_strategy = "server_error"
                
                if should_retry:
                    if retry_strategy == "rate_limit":
                        # Exponential backoff with jitter for rate limits
                        wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                        print(f"API rate limit hit (429), waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    elif retry_strategy == "server_error":
                        # Shorter backoff for server errors with jitter
                        wait_time = (base_delay * (1.5 ** attempt)) + random.uniform(0.5, 1.5)
                        error_type = {500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout"}.get(status_code, f"Server Error {status_code}")
                        print(f"Google Sheets API {error_type} ({status_code}), waiting {wait_time:.1f}s before retry {attempt + 1}/2")
                    
                    time.sleep(wait_time)
                else:
                    # Log the error type for debugging
                    if status_code:
                        error_type = {
                            429: "Rate Limit Exceeded", 
                            500: "Internal Server Error", 
                            502: "Bad Gateway", 
                            503: "Service Unavailable", 
                            504: "Gateway Timeout"
                        }.get(status_code, f"HTTP Error {status_code}")
                        print(f"API call failed with {error_type} ({status_code}). Max retries reached.")
                    
                    # Re-raise the original exception
                    raise e
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def _add_api_delay(self, delay=0.2):
        """Add small delay between API calls to prevent server overload"""
        time.sleep(delay)
        
    def _authenticate(self, service_account_json=None):
        """Authenticate with Google Sheets API"""
        try:
            scope = ['https://www.googleapis.com/auth/spreadsheets']
            
            if service_account_json:
                # Use provided JSON string (for GitHub Actions)
                service_account_info = json.loads(service_account_json)
                creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
            else:
                # Check for GOOGLE_SERVICE_ACCOUNT_JSON (GitHub Actions)
                service_account_json_env = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
                if service_account_json_env:
                    # Check if it's a file path or JSON content
                    if service_account_json_env.startswith('/') or service_account_json_env.endswith('.json'):
                        # It's a file path - for local development
                        creds = Credentials.from_service_account_file(service_account_json_env, scopes=scope)
                    else:
                        # It's JSON content - for GitHub Actions
                        service_account_info = json.loads(service_account_json_env)
                        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
                else:
                    # Last fallback to local file (for development)
                    service_account_file = "pullus-pipeline-40a5302e034d.json"
                    creds = Credentials.from_service_account_file(service_account_file, scopes=scope)
            
            return gspread.authorize(creds)
        except Exception as e:
            print("Error: Authentication failed. Please check your credentials configuration.")
            raise e
    
    def _load_purchase_targets(self):
        """Load 2026 purchase targets from env var (production) or local JSON file."""
        targets_json = os.getenv('PURCHASE_TARGETS_2026')
        if targets_json:
            raw = json.loads(targets_json)
        else:
            targets_path = os.path.join(os.path.dirname(__file__), 'purchase_targets_2026.json')
            if os.path.exists(targets_path):
                with open(targets_path, 'r') as f:
                    raw = json.load(f)
            else:
                print("Warning: No purchase targets found (env var PURCHASE_TARGETS_2026 or purchase_targets_2026.json)")
                return {}
        return {int(k): v for k, v in raw.items()}

    def read_sheet_data(self, sheet_id, sheet_name="Sheet1"):
        """Read data from Google Sheet with optimized error handling"""
        try:
            sheet = self._api_call_with_retry(lambda: self.gc.open_by_key(sheet_id))
            self._add_api_delay()
            
            worksheet = self._api_call_with_retry(lambda: sheet.worksheet(sheet_name))
            self._add_api_delay()
            
            # Get all values and skip first 3 rows - use field optimization
            all_values = self._api_call_with_retry(lambda: worksheet.get_all_values())
            if len(all_values) > 3:
                headers = all_values[3]  # Row 4 contains headers
                data_rows = all_values[4:]  # Data starts from row 5
                
                # Create DataFrame with headers
                df = pd.DataFrame(data_rows, columns=headers)
                # Remove empty rows
                df = df.dropna(how='all')
                
                # Convert numeric columns to numeric values
                numeric_columns = (['NUMBER OF BIRDS']
                                   + [p['purchase_col'] for p in PRODUCTS]
                                   + [p['inventory_col'] for p in PRODUCTS])
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading sheet: {e}")
            return pd.DataFrame()
    
    def process_purchase_data(self, purchase_df):
        """Process purchase data - sum by day and purchase officer"""
        if purchase_df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        purchase_df['DATE'] = pd.to_datetime(purchase_df['DATE'])
        
        # Strip whitespace from purchase officer names
        purchase_df['PURCHASE OFFICER NAME'] = purchase_df['PURCHASE OFFICER NAME'].str.strip()
        
        # Group by date and purchase officer, sum the metrics
        agg_spec = {'NUMBER OF BIRDS': 'sum'}
        for p in PRODUCTS:
            if p['purchase_col'] in purchase_df.columns:
                agg_spec[p['purchase_col']] = 'sum'
        agg_spec['INVOICE NUMBER'] = lambda x: list(x)  # Collect all invoice numbers
        grouped = purchase_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg(agg_spec).reset_index()

        return grouped
    
    def process_inventory_data(self, inventory_df):
        """Process inventory data - sum by day and purchase officer"""
        if inventory_df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        inventory_df['DATE'] = pd.to_datetime(inventory_df['DATE'])
        
        # Strip whitespace from purchase officer names
        inventory_df['PURCHASE OFFICER NAME'] = inventory_df['PURCHASE OFFICER NAME'].str.strip()
        
        # Split comma-separated invoice numbers (handle various spacing patterns)
        def split_invoices(invoice_str):
            if pd.isna(invoice_str) or invoice_str == '':
                return []
            # Split by comma and strip whitespace, filter out empty strings
            return [inv.strip() for inv in str(invoice_str).split(',') if inv.strip()]
        
        inventory_df['INVOICE_LIST'] = inventory_df['INVOICE NUMBER'].apply(split_invoices)
        
        # Group by date and purchase officer, sum the metrics and collect invoice numbers
        agg_spec = {'NUMBER OF BIRDS': 'sum'}
        for p in PRODUCTS:
            if p['inventory_col'] in inventory_df.columns:
                agg_spec[p['inventory_col']] = 'sum'
        agg_spec['INVOICE_LIST'] = lambda x: [item for sublist in x for item in sublist]  # Flatten all invoice lists
        grouped = inventory_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg(agg_spec).reset_index()
        
        # Reconstruct the INVOICE NUMBER column from the flattened list
        grouped['INVOICE NUMBER'] = grouped['INVOICE_LIST'].apply(lambda x: ','.join(sorted(set(x))))
        
        return grouped
    
    def generate_weight_discrepancy_report(self, purchase_grouped, inventory_df):
        """Generate weight and bird count discrepancy report"""
        discrepancies = []
        processed_combinations = set()
        
        # Process purchase records and find matches in inventory
        for _, purchase_row in purchase_grouped.iterrows():
            date = purchase_row['DATE']
            officer = purchase_row['PURCHASE OFFICER NAME']
            combination = (date, officer)
            processed_combinations.add(combination)
            
            # Find matching inventory record
            inventory_match = inventory_df[
                (inventory_df['DATE'] == date) & 
                (inventory_df['PURCHASE OFFICER NAME'] == officer)
            ]
            
            if not inventory_match.empty:
                inv_row = inventory_match.iloc[0]

                # Bird discrepancy (round to 2 decimal places)
                bird_diff = round(inv_row['NUMBER OF BIRDS'] - purchase_row['NUMBER OF BIRDS'], 2)
                bird_pct_diff = round((bird_diff / purchase_row['NUMBER OF BIRDS']) * 100, 2) if purchase_row['NUMBER OF BIRDS'] > 0 else 0

                # Per-product weight discrepancies (purchase as baseline)
                tolerance = 0.01  # 0.01 tolerance for floating point precision
                has_discrepancy = abs(bird_diff) > tolerance
                weight_cells = {}
                pct_cells = {}
                for p in self.active_products:
                    label = p['label']
                    pw = purchase_row.get(p['purchase_col'], 0) or 0
                    iw = inv_row.get(p['inventory_col'], 0) or 0
                    diff = round(iw - pw, 2)
                    pct = round((diff / pw) * 100, 2) if pw > 0 else 0
                    if abs(diff) > tolerance:
                        has_discrepancy = True
                    weight_cells[f'Purchase {label} Weight'] = f"{round(pw, 2):,}"
                    weight_cells[f'Inventory {label} Weight'] = f"{round(iw, 2):,}"
                    weight_cells[f'{label} Weight Difference'] = f"{diff:,}"
                    pct_cells[f'{label} Weight Percentage Difference'] = f"{pct}%"

                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': f"{round(purchase_row['NUMBER OF BIRDS']):,.0f}",
                    'Inventory Birds': f"{round(inv_row['NUMBER OF BIRDS']):,.0f}",
                    'Birds Difference': f"{bird_diff:,.0f}",
                    **weight_cells,
                    'Birds Percentage Difference': f"{bird_pct_diff}%",
                    **pct_cells,
                    'Status': 'Discrepancy' if has_discrepancy else 'Match',
                    'Responsible Party': '',
                    'Root Cause': '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING' if has_discrepancy else 'RESOLVED',
                    'Resolution Date': ''
                })
            else:
                # No inventory record found - purchase exists but not in inventory
                weight_cells = {}
                pct_cells = {}
                for p in self.active_products:
                    label = p['label']
                    pw = purchase_row.get(p['purchase_col'], 0) or 0
                    weight_cells[f'Purchase {label} Weight'] = f"{round(pw, 2):,}"
                    weight_cells[f'Inventory {label} Weight'] = 'NOT FOUND'
                    weight_cells[f'{label} Weight Difference'] = 'N/A'
                    pct_cells[f'{label} Weight Percentage Difference'] = 'N/A'
                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': f"{round(purchase_row['NUMBER OF BIRDS']):,.0f}",
                    'Inventory Birds': 'NOT FOUND',
                    'Birds Difference': 'N/A',
                    **weight_cells,
                    'Birds Percentage Difference': 'N/A',
                    **pct_cells,
                    'Status': 'MISSING IN INVENTORY',
                    'Responsible Party': '',
                    'Root Cause': '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING',
                    'Resolution Date': ''
                })
        
        # Now check for inventory records that don't exist in purchase
        for _, inv_row in inventory_df.iterrows():
            date = inv_row['DATE']
            officer = inv_row['PURCHASE OFFICER NAME']
            combination = (date, officer)
            
            if combination not in processed_combinations:
                # Inventory record exists but not in purchase
                weight_cells = {}
                pct_cells = {}
                for p in self.active_products:
                    label = p['label']
                    iw = inv_row.get(p['inventory_col'], 0) or 0
                    weight_cells[f'Purchase {label} Weight'] = 'NOT FOUND'
                    weight_cells[f'Inventory {label} Weight'] = f"{round(iw, 2):,}"
                    weight_cells[f'{label} Weight Difference'] = 'N/A'
                    pct_cells[f'{label} Weight Percentage Difference'] = 'N/A'
                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': 'NOT FOUND',
                    'Inventory Birds': f"{round(inv_row['NUMBER OF BIRDS']):,.0f}",
                    'Birds Difference': 'N/A',
                    **weight_cells,
                    'Birds Percentage Difference': 'N/A',
                    **pct_cells,
                    'Status': 'MISSING IN PURCHASE',
                    'Responsible Party': '',
                    'Root Cause': '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING',
                    'Resolution Date': ''
                })

        # Convert to DataFrame and sort by date chronologically
        df = pd.DataFrame(discrepancies)
        if not df.empty:
            # Create temporary datetime column for sorting
            df['Date_Sort'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
            df = df.sort_values('Date_Sort').drop('Date_Sort', axis=1).reset_index(drop=True)

        return df
    
    def generate_invoice_mismatch_report(self, purchase_grouped, inventory_df):
        """Generate invoice mismatch report"""
        mismatches = []
        processed_combinations = set()
        
        # Process purchase records and find matches in inventory
        for _, purchase_row in purchase_grouped.iterrows():
            date = purchase_row['DATE']
            officer = purchase_row['PURCHASE OFFICER NAME']
            combination = (date, officer)
            processed_combinations.add(combination)
            purchase_invoices = set(map(str, purchase_row['INVOICE NUMBER']))
            
            # Find matching inventory record
            inventory_match = inventory_df[
                (inventory_df['DATE'] == date) & 
                (inventory_df['PURCHASE OFFICER NAME'] == officer)
            ]
            
            if not inventory_match.empty:
                inv_row = inventory_match.iloc[0]
                inventory_invoices = set(map(str, inv_row['INVOICE_LIST']))
                
                # Find mismatches
                missing_in_inventory = purchase_invoices - inventory_invoices
                extra_in_inventory = inventory_invoices - purchase_invoices
                
                if missing_in_inventory or extra_in_inventory:
                    mismatches.append({
                        'Date': date.strftime('%d-%b-%Y'),
                        'Purchase Officer': officer,
                        'Purchase Invoices': ', '.join(sorted(purchase_invoices)),
                        'Inventory Invoices': ', '.join(sorted(inventory_invoices)),
                        'Missing in Inventory': ', '.join(sorted(missing_in_inventory)) if missing_in_inventory else 'None',
                        'Extra in Inventory': ', '.join(sorted(extra_in_inventory)) if extra_in_inventory else 'None',
                        'Status': 'MISMATCH',
                        'Responsible Party': '',
                        'Root Cause': '',
                        'Purchase Team Comments': '',
                        'Inventory Team Comments': '',
                        'Resolution Status': 'PENDING',
                        'Resolution Date': ''
                        })
                else:
                    mismatches.append({
                        'Date': date.strftime('%d-%b-%Y'),
                        'Purchase Officer': officer,
                        'Purchase Invoices': ', '.join(sorted(purchase_invoices)),
                        'Inventory Invoices': ', '.join(sorted(inventory_invoices)),
                        'Missing in Inventory': 'None',
                        'Extra in Inventory': 'None',
                        'Status': 'MATCH',
                        'Responsible Party': '',
                        'Root Cause': '',
                        'Purchase Team Comments': '',
                        'Inventory Team Comments': '',
                        'Resolution Status': 'RESOLVED',
                        'Resolution Date': ''
                        })
            else:
                # No inventory record found - purchase exists but not in inventory
                mismatches.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Invoices': ', '.join(sorted(purchase_invoices)),
                    'Inventory Invoices': 'NOT FOUND',
                    'Missing in Inventory': 'ALL',
                    'Extra in Inventory': 'N/A',
                    'Status': 'MISSING IN INVENTORY',
                    'Responsible Party': '',
                    'Root Cause': '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING',
                    'Resolution Date': ''
                })
        
        # Now check for inventory records that don't exist in purchase
        for _, inv_row in inventory_df.iterrows():
            date = inv_row['DATE']
            officer = inv_row['PURCHASE OFFICER NAME']
            combination = (date, officer)
            
            if combination not in processed_combinations:
                # Inventory record exists but not in purchase
                inventory_invoices = set(map(str, inv_row['INVOICE_LIST']))
                
                mismatches.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Invoices': 'NOT FOUND',
                    'Inventory Invoices': ', '.join(sorted(inventory_invoices)),
                    'Missing in Inventory': 'N/A',
                    'Extra in Inventory': 'ALL',
                    'Status': 'MISSING IN PURCHASE',
                    'Responsible Party': '',
                    'Root Cause': '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING',
                    'Resolution Date': ''
                })

        # Convert to DataFrame and sort by date chronologically
        df = pd.DataFrame(mismatches)
        if not df.empty:
            # Create temporary datetime column for sorting
            df['Date_Sort'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
            df = df.sort_values('Date_Sort').drop('Date_Sort', axis=1).reset_index(drop=True)

        return df
    
    def generate_monthly_summary_report(self, purchase_grouped, inventory_df):
        """Generate monthly summary report by purchase officer"""
        summaries = []
        
        # Process purchase data by month and officer
        if not purchase_grouped.empty:
            purchase_grouped['YEAR_MONTH'] = purchase_grouped['DATE'].dt.to_period('M')
            # Named aggregation keeps columns flat (no multi-level index to flatten)
            agg_named = {
                'NUMBER OF BIRDS': ('NUMBER OF BIRDS', 'sum'),
                'OFFTAKE_COUNT': ('NUMBER OF BIRDS', 'count'),  # count gives us offtake count
            }
            for p in PRODUCTS:
                if p['purchase_col'] in purchase_grouped.columns:
                    agg_named[p['purchase_col']] = (p['purchase_col'], 'sum')
            purchase_monthly = purchase_grouped.groupby(['YEAR_MONTH', 'PURCHASE OFFICER NAME']).agg(**agg_named).reset_index()
        else:
            purchase_monthly = pd.DataFrame()
        
        # Process inventory data by month and officer
        if not inventory_df.empty:
            inventory_df['YEAR_MONTH'] = inventory_df['DATE'].dt.to_period('M')
            agg_named = {'NUMBER OF BIRDS': ('NUMBER OF BIRDS', 'sum')}
            for p in PRODUCTS:
                if p['inventory_col'] in inventory_df.columns:
                    agg_named[p['inventory_col']] = (p['inventory_col'], 'sum')
            inventory_monthly = inventory_df.groupby(['YEAR_MONTH', 'PURCHASE OFFICER NAME']).agg(**agg_named).reset_index()
        else:
            inventory_monthly = pd.DataFrame()
        
        # Get all unique combinations of year_month and officer
        all_combinations = set()
        if not purchase_monthly.empty:
            for _, row in purchase_monthly.iterrows():
                all_combinations.add((row['YEAR_MONTH'], row['PURCHASE OFFICER NAME']))
        if not inventory_monthly.empty:
            for _, row in inventory_monthly.iterrows():
                all_combinations.add((row['YEAR_MONTH'], row['PURCHASE OFFICER NAME']))
        
        # Generate summary for each combination
        for year_month, officer in sorted(all_combinations):
            # Get purchase data for this combination
            purchase_match = purchase_monthly[
                (purchase_monthly['YEAR_MONTH'] == year_month) & 
                (purchase_monthly['PURCHASE OFFICER NAME'] == officer)
            ]
            
            # Get inventory data for this combination
            inventory_match = inventory_monthly[
                (inventory_monthly['YEAR_MONTH'] == year_month) & 
                (inventory_monthly['PURCHASE OFFICER NAME'] == officer)
            ]
            
            # Extract bird values or set to 0 if not found
            if not purchase_match.empty:
                p_birds = round(purchase_match.iloc[0]['NUMBER OF BIRDS'], 2)
                offtake_count = int(purchase_match.iloc[0]['OFFTAKE_COUNT'])
                p_row = purchase_match.iloc[0]
            else:
                p_birds = 0
                offtake_count = 0
                p_row = None

            if not inventory_match.empty:
                i_birds = round(inventory_match.iloc[0]['NUMBER OF BIRDS'], 2)
                i_row = inventory_match.iloc[0]
            else:
                i_birds = 0
                i_row = None

            # Bird differences
            birds_diff = round(i_birds - p_birds, 2)
            birds_pct = round((birds_diff / p_birds) * 100, 2) if p_birds > 0 else 0

            summary = {
                'Month': str(year_month),
                'Purchase Officer': officer,
                'Offtake Count': f"{offtake_count:,}",
                'Purchase Birds Total': f"{p_birds:,.0f}",
                'Inventory Birds Total': f"{i_birds:,.0f}",
                'Birds Difference': f"{birds_diff:,.0f}",
                'Birds Percentage Difference': f"{birds_pct}%",
            }
            # Per-product weight totals, differences and percentages
            for p in self.active_products:
                label = p['label']
                pw = round(p_row.get(p['purchase_col'], 0), 2) if p_row is not None else 0
                iw = round(i_row.get(p['inventory_col'], 0), 2) if i_row is not None else 0
                diff = round(iw - pw, 2)
                pct = round((diff / pw) * 100, 2) if pw > 0 else 0
                summary[f'Purchase {label} Weight Total'] = f"{pw:,.2f}"
                summary[f'Inventory {label} Weight Total'] = f"{iw:,.2f}"
                summary[f'{label} Weight Difference'] = f"{diff:,.2f}"
                summary[f'{label} Weight Percentage Difference'] = f"{pct}%"
            summaries.append(summary)
        
        # Group summaries by month and add grand totals per month
        summary_df = pd.DataFrame(summaries)
        if not summary_df.empty:
            # Group by month and reorganize with monthly grand totals
            final_summaries = []
            
            # Get unique months in order
            unique_months = summary_df['Month'].unique()
            
            for month in unique_months:
                # Add all officer rows for this month
                month_rows = summary_df[summary_df['Month'] == month]
                for _, row in month_rows.iterrows():
                    final_summaries.append(row.to_dict())
                
                # Calculate and add grand total for this month
                month_grand_total = self._calculate_monthly_grand_total(month_rows, month)
                final_summaries.append(month_grand_total)
            
            return pd.DataFrame(final_summaries)
        
        return pd.DataFrame(summaries)
    
    def generate_purchase_target_report(self, purchase_grouped):
        """Generate 2026 purchase target tracker comparing actuals vs targets."""
        targets = self.purchase_targets_2026
        if not targets:
            print("Skipping purchase target report - no targets loaded")
            return pd.DataFrame()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Get 2026 actuals grouped by month
        actuals = {}
        if not purchase_grouped.empty:
            data_2026 = purchase_grouped[purchase_grouped['DATE'].dt.year == 2026].copy()
            if not data_2026.empty:
                data_2026['MONTH'] = data_2026['DATE'].dt.month
                monthly = data_2026.groupby('MONTH').agg({
                    'NUMBER OF BIRDS': 'sum',
                    'PURCHASED CHICKEN WEIGHT': 'sum'
                }).to_dict('index')
                actuals = monthly

        rows = []
        cumulative_birds_deficit = 0
        cumulative_vol_deficit = 0

        for q in range(4):
            q_target_birds = 0
            q_actual_birds = 0
            q_target_vol = 0
            q_actual_vol = 0
            q_birds_deficit = 0
            q_vol_deficit = 0
            q_start_birds_deficit = cumulative_birds_deficit
            q_start_vol_deficit = cumulative_vol_deficit

            for m_in_q in range(3):
                month_num = q * 3 + m_in_q + 1
                t = targets.get(month_num, {'birds': 0, 'vol_kg': 0})
                target_birds = t['birds']
                target_vol = t['vol_kg']

                actual_birds = round(actuals.get(month_num, {}).get('NUMBER OF BIRDS', 0), 0)
                actual_vol = round(actuals.get(month_num, {}).get('PURCHASED CHICKEN WEIGHT', 0), 2)

                birds_deficit = target_birds - actual_birds
                vol_deficit = target_vol - actual_vol

                adjusted_birds = target_birds + cumulative_birds_deficit
                adjusted_vol = target_vol + cumulative_vol_deficit

                achievement = round((actual_birds / adjusted_birds) * 100, 1) if adjusted_birds > 0 else 0

                rows.append({
                    'Period': month_names[month_num - 1],
                    'Target Birds': f"{target_birds:,}",
                    'Adjusted Birds Target': f"{int(adjusted_birds):,}",
                    'Actual Birds': f"{int(actual_birds):,}",
                    'Birds Deficit': f"{int(birds_deficit):,}",
                    'Cumulative Birds Deficit': f"{int(cumulative_birds_deficit + birds_deficit):,}",
                    'Target Vol (Kg)': f"{target_vol:,.2f}",
                    'Adjusted Vol Target (Kg)': f"{adjusted_vol:,.2f}",
                    'Actual Vol (Kg)': f"{actual_vol:,.2f}",
                    'Vol Deficit (Kg)': f"{vol_deficit:,.2f}",
                    'Cumulative Vol Deficit (Kg)': f"{cumulative_vol_deficit + vol_deficit:,.2f}",
                    'Achievement %': f"{achievement}%"
                })

                cumulative_birds_deficit += birds_deficit
                cumulative_vol_deficit += vol_deficit

                q_target_birds += target_birds
                q_actual_birds += actual_birds
                q_target_vol += target_vol
                q_actual_vol += actual_vol
                q_birds_deficit += birds_deficit
                q_vol_deficit += vol_deficit

            adjusted_q_target_birds = q_target_birds + q_start_birds_deficit
            adjusted_q_target_vol = q_target_vol + q_start_vol_deficit
            q_achievement = round((q_actual_birds / adjusted_q_target_birds) * 100, 1) if adjusted_q_target_birds > 0 else 0
            rows.append({
                'Period': f"Q{q + 1} TOTAL",
                'Target Birds': f"{int(q_target_birds):,}",
                'Adjusted Birds Target': f"{int(adjusted_q_target_birds):,}",
                'Actual Birds': f"{int(q_actual_birds):,}",
                'Birds Deficit': f"{int(q_birds_deficit):,}",
                'Cumulative Birds Deficit': f"{int(cumulative_birds_deficit):,}",
                'Target Vol (Kg)': f"{q_target_vol:,.2f}",
                'Adjusted Vol Target (Kg)': f"{adjusted_q_target_vol:,.2f}",
                'Actual Vol (Kg)': f"{q_actual_vol:,.2f}",
                'Vol Deficit (Kg)': f"{q_vol_deficit:,.2f}",
                'Cumulative Vol Deficit (Kg)': f"{cumulative_vol_deficit:,.2f}",
                'Achievement %': f"{q_achievement}%"
            })

            # Spacing row after each quarter (except last)
            if q < 3:
                rows.append({col: '' for col in ['Period', 'Target Birds', 'Adjusted Birds Target',
                    'Actual Birds', 'Birds Deficit', 'Cumulative Birds Deficit',
                    'Target Vol (Kg)', 'Adjusted Vol Target (Kg)', 'Actual Vol (Kg)',
                    'Vol Deficit (Kg)', 'Cumulative Vol Deficit (Kg)', 'Achievement %']})

        # Spacing before annual total
        rows.append({col: '' for col in rows[0].keys()})

        # Annual total
        annual_target_birds = sum(targets[m]['birds'] for m in range(1, 13))
        annual_target_vol = sum(targets[m]['vol_kg'] for m in range(1, 13))
        annual_actual_birds = sum(actuals.get(m, {}).get('NUMBER OF BIRDS', 0) for m in range(1, 13))
        annual_actual_vol = sum(actuals.get(m, {}).get('PURCHASED CHICKEN WEIGHT', 0) for m in range(1, 13))
        annual_birds_deficit = annual_target_birds - annual_actual_birds
        annual_vol_deficit = annual_target_vol - annual_actual_vol
        annual_achievement = round((annual_actual_birds / annual_target_birds) * 100, 1) if annual_target_birds > 0 else 0

        rows.append({
            'Period': 'ANNUAL TOTAL',
            'Target Birds': f"{int(annual_target_birds):,}",
            'Adjusted Birds Target': f"{int(annual_target_birds):,}",
            'Actual Birds': f"{int(annual_actual_birds):,}",
            'Birds Deficit': f"{int(annual_birds_deficit):,}",
            'Cumulative Birds Deficit': f"{int(cumulative_birds_deficit):,}",
            'Target Vol (Kg)': f"{annual_target_vol:,.2f}",
            'Adjusted Vol Target (Kg)': f"{annual_target_vol:,.2f}",
            'Actual Vol (Kg)': f"{annual_actual_vol:,.2f}",
            'Vol Deficit (Kg)': f"{annual_vol_deficit:,.2f}",
            'Cumulative Vol Deficit (Kg)': f"{cumulative_vol_deficit:,.2f}",
            'Achievement %': f"{annual_achievement}%"
        })

        return pd.DataFrame(rows)

    def generate_purchase_officer_performance_report(self, purchase_grouped):
        """Generate purchase officer performance report with averages and totals"""
        if purchase_grouped.empty:
            return pd.DataFrame()

        # Products with data that also have a column present
        present_products = [p for p in self.active_products if p['purchase_col'] in purchase_grouped.columns]

        # Calculate averages and sums per purchase officer (named agg keeps columns flat)
        agg_named = {
            'Average Birds per Day': ('NUMBER OF BIRDS', 'mean'),
            'Total Purchase Days': ('NUMBER OF BIRDS', 'count'),
            'Total Birds': ('NUMBER OF BIRDS', 'sum'),
        }
        for p in present_products:
            agg_named[f"Average {p['label']} Weight per Day"] = (p['purchase_col'], 'mean')
            agg_named[f"Total {p['label']} Weight"] = (p['purchase_col'], 'sum')
        performance_stats = (purchase_grouped.groupby('PURCHASE OFFICER NAME')
                             .agg(**agg_named).reset_index()
                             .rename(columns={'PURCHASE OFFICER NAME': 'Purchase Officer'}))

        # Round to appropriate decimal places
        performance_stats['Average Birds per Day'] = performance_stats['Average Birds per Day'].round(0)
        performance_stats['Total Birds'] = performance_stats['Total Birds'].round(0)
        for p in present_products:
            performance_stats[f"Average {p['label']} Weight per Day"] = performance_stats[f"Average {p['label']} Weight per Day"].round(2)
            performance_stats[f"Total {p['label']} Weight"] = performance_stats[f"Total {p['label']} Weight"].round(2)

        # Calculate combined total across all products
        total_cols = [f"Total {p['label']} Weight" for p in present_products]
        performance_stats['Combined Total'] = (performance_stats[total_cols].sum(axis=1).round(2)
                                               if total_cols else 0.0)

        # Calculate data-driven performance thresholds based on Combined Total weight
        q75 = float(performance_stats['Combined Total'].quantile(0.75))
        q50 = float(performance_stats['Combined Total'].quantile(0.50))
        q25 = float(performance_stats['Combined Total'].quantile(0.25))

        # Format for display - averages first (birds then products), then totals, then combined
        performance_report = []
        for _, row in performance_stats.iterrows():
            entry = {
                'Purchase Officer': row['Purchase Officer'],
                'Average Birds per Day': f"{row['Average Birds per Day']:,.0f}",
            }
            for p in present_products:
                avg_col = f"Average {p['label']} Weight per Day"
                entry[f"{avg_col} (kg)"] = f"{row[avg_col]:,.2f}"
            entry['Total Purchase Days'] = f"{int(row['Total Purchase Days']):,}"
            entry['Total Birds'] = f"{row['Total Birds']:,.0f}"
            for p in present_products:
                tot_col = f"Total {p['label']} Weight"
                entry[f"{tot_col} (kg)"] = f"{row[tot_col]:,.2f}"
            entry['Combined Total (kg)'] = f"{row['Combined Total']:,.2f}"
            entry['Volume Category'] = self._calculate_data_driven_performance_rating(
                row['Combined Total'], q75, q50, q25
            )
            performance_report.append(entry)

        # Sort by combined total weight (descending)
        performance_df = pd.DataFrame(performance_report)
        performance_df = performance_df.sort_values('Combined Total (kg)',
                                                   key=lambda x: x.str.replace(',', '').astype(float),
                                                   ascending=False).reset_index(drop=True)

        # Add grand total row
        grand_total_row = self._calculate_performance_grand_total(performance_df)
        performance_df = pd.concat([performance_df, pd.DataFrame([grand_total_row])], ignore_index=True)

        return performance_df
    
    def _calculate_data_driven_performance_rating(self, total_weight, q75, q50, q25):
        """Calculate volume category based on combined total weight percentiles"""
        if total_weight >= q75:
            return "Highest Volume"      # Top 25%
        elif total_weight >= q50:
            return "High Volume"         # Above median (50th-75th percentile)
        elif total_weight >= q25:
            return "Moderate Volume"     # Below median but above bottom 25%
        else:
            return "Lower Volume"        # Bottom 25%

    @staticmethod
    def _parse_numeric(value_str):
        """Parse a comma-formatted number string back to float, 0 on failure."""
        try:
            return float(str(value_str).replace(',', ''))
        except (ValueError, AttributeError):
            return 0

    @staticmethod
    def _format_weight(weight, use_tonnes=False):
        """Format a kg weight. With use_tonnes=True, switch to tonnes above 1000kg."""
        if use_tonnes and abs(weight) >= 1000:
            tonnes_value = weight / 1000
            unit = "tonne" if abs(tonnes_value) == 1.00 else "tonnes"
            return f"{tonnes_value:,.2f} {unit}"
        return f"{weight:,.2f} kg"

    def _calculate_performance_grand_total(self, performance_df):
        """Calculate grand total for performance report"""
        # Calculate totals for the sum columns
        total_birds = sum(self._parse_numeric(val) for val in performance_df['Total Birds'])
        total_combined = sum(self._parse_numeric(val) for val in performance_df['Combined Total (kg)'])

        # Build grand total in the same column order as the report rows
        grand = {
            'Purchase Officer': '═══════ GRAND TOTAL ═══════',
            'Average Birds per Day': '',
        }
        for p in self.active_products:
            avg_col = f"Average {p['label']} Weight per Day (kg)"
            if avg_col in performance_df.columns:
                grand[avg_col] = ''
        grand['Total Purchase Days'] = ''
        grand['Total Birds'] = f"{total_birds:,.0f}"
        for p in self.active_products:
            tot_col = f"Total {p['label']} Weight (kg)"
            if tot_col in performance_df.columns:
                total_w = sum(self._parse_numeric(val) for val in performance_df[tot_col])
                grand[tot_col] = self._format_weight(total_w, use_tonnes=True)
        grand['Combined Total (kg)'] = self._format_weight(total_combined, use_tonnes=True)
        grand['Volume Category'] = ''
        return grand

    def _calculate_monthly_grand_total(self, month_df, month):
        """Calculate grand total for a specific month"""
        # Calculate totals for each category for this month
        total_offtake = sum(int(self._parse_numeric(val)) for val in month_df['Offtake Count'])
        total_p_birds = sum(self._parse_numeric(val) for val in month_df['Purchase Birds Total'])
        total_i_birds = sum(self._parse_numeric(val) for val in month_df['Inventory Birds Total'])
        total_birds_diff = total_i_birds - total_p_birds
        total_birds_pct = round((total_birds_diff / total_p_birds) * 100, 2) if total_p_birds > 0 else 0

        grand = {
            'Month': '',
            'Purchase Officer': f'═══════ {month} GRAND TOTAL ═══════',
            'Offtake Count': f"{total_offtake:,}",
            'Purchase Birds Total': f"{total_p_birds:,.0f}",
            'Inventory Birds Total': f"{total_i_birds:,.0f}",
            'Birds Difference': f"{total_birds_diff:,.0f}",
            'Birds Percentage Difference': f"{total_birds_pct}%",
        }
        # Per-product totals (only products with data, matching the officer rows).
        # Monthly grand totals always use kg (no tonnes conversion).
        for p in self.active_products:
            label = p['label']
            p_col = f'Purchase {label} Weight Total'
            i_col = f'Inventory {label} Weight Total'
            total_p = sum(self._parse_numeric(val) for val in month_df[p_col]) if p_col in month_df.columns else 0
            total_i = sum(self._parse_numeric(val) for val in month_df[i_col]) if i_col in month_df.columns else 0
            diff = total_i - total_p
            pct = round((diff / total_p) * 100, 2) if total_p > 0 else 0
            grand[p_col] = self._format_weight(total_p)
            grand[i_col] = self._format_weight(total_i)
            grand[f'{label} Weight Difference'] = self._format_weight(diff)
            grand[f'{label} Weight Percentage Difference'] = f"{pct}%"
        return grand
    
    def update_google_sheet_with_preservation(self, sheet_id, sheet_name, df, title, sheet_type):
        """Update Google Sheet with data preservation and optimized error handling"""
        # Separate title and timestamp for better column width management
        timestamp = self._get_wat_timestamp()
        timestamp_row = f"Last Updated: {timestamp}"
        try:
            sheet = self._api_call_with_retry(lambda: self.gc.open_by_key(sheet_id))
            self._add_api_delay()
            
            # Try to get existing worksheet, create if doesn't exist
            try:
                worksheet = self._api_call_with_retry(lambda: sheet.worksheet(sheet_name))
            except:
                worksheet = self._api_call_with_retry(lambda: sheet.add_worksheet(title=sheet_name, rows=1000, cols=20))
                self._add_api_delay()
            
            # Make a copy of the dataframe to avoid modifying original
            df_copy = df.copy()
            
            # Preserve existing data with optimized API calls
            df_copy = self._preserve_existing_data(worksheet, df_copy, sheet_type)
            
            # Customize columns for the specific sheet
            df_copy = self._customize_columns_for_sheet(df_copy, sheet_type)
            
            # Clear the worksheet content and formatting with delay
            self._api_call_with_retry(lambda: worksheet.clear())
            self._add_api_delay(0.5)  # Longer delay after clear operation
            
            # Clear all formatting for entire sheet
            try:
                # Clear formatting for entire sheet (not just current data range) to remove old formatting
                self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                    'requests': [{
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet.id
                                # Not specifying row/column limits clears the entire sheet
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                    'textFormat': {
                                        'fontFamily': 'Lato',
                                        'fontSize': 10,
                                        'bold': False
                                    },
                                    'horizontalAlignment': 'LEFT'
                                }
                            },
                            'fields': 'userEnteredFormat'
                        }
                    }]
                }))
                self._add_api_delay(0.3)  # Delay after formatting clear
            except Exception as e:
                print(f"Warning: Could not clear formatting: {e}")
            
            # Prepare data with separate title and timestamp rows - optimize data preparation
            data_to_write = []
            data_to_write.append([title])  # Title row
            data_to_write.append([timestamp_row])  # Timestamp row

            # Add explainer for purchase target tracker
            if sheet_type == 'target_tracker':
                data_to_write.append([])  # Empty row
                data_to_write.append(['HOW TO READ THIS REPORT:'])
                data_to_write.append(['Birds Deficit / Vol Deficit = Target minus Actual. Red text means behind target, green text means ahead.'])
                data_to_write.append(['Cumulative Deficit = Running total of all deficits from Jan to current month. Shows how far behind (or ahead) overall.'])
                data_to_write.append(['Adjusted Target = Original monthly target + previous cumulative deficit. This is what you NEED to buy this month to get back on track.'])
                data_to_write.append(['Quarterly / Annual Adjusted Target = period target + deficit carried INTO the period. NOT the sum of monthly Adjusted Targets (that would double-count deficits).'])
                data_to_write.append(['For Q1 and Annual there is no prior deficit, so Adjusted equals the raw Target — shown for consistency with the other rows.'])
                data_to_write.append(['Achievement % = Actual Birds divided by Adjusted Target. Shows progress against what is needed to catch up, not just the base target.'])

            # Add Volume Category explainer for performance reports
            elif sheet_type == 'performance':
                data_to_write.append([])  # Empty row
                data_to_write.append(['HOW VOLUME CATEGORIES WORK:'])
                data_to_write.append(['We rank purchase officers based on their Combined Total weight (all product weights added together).'])
                data_to_write.append(['Then we divide them into 4 groups:'])
                data_to_write.append(['Highest Volume = Top 25% (the officers who handled the most weight)'])
                data_to_write.append(['High Volume = Next 25% (above average, but not the highest)'])
                data_to_write.append(['Moderate Volume = Next 25% (below average, but not the lowest)'])
                data_to_write.append(['Lower Volume = Bottom 25% (the officers who handled the least weight)'])
                data_to_write.append(['This ranking compares all officers against each other, so the categories change as performance changes.'])

            data_to_write.append([])  # Empty row
            data_to_write.append(df_copy.columns.tolist())  # Headers
            
            # Add data rows with optimized processing
            for row in df_copy.values:
                # Convert numpy types to Python types for JSON serialization
                row_data = []
                for val in row:
                    if pd.isna(val):
                        row_data.append('')
                    elif isinstance(val, (pd.Int64Dtype, pd.Float64Dtype)) or hasattr(val, 'dtype'):
                        # Round to 2 decimal places for numeric values
                        numeric_val = float(val) if pd.notna(val) else 0
                        row_data.append(round(numeric_val, 2))
                    else:
                        row_data.append(str(val))
                data_to_write.append(row_data)
            
            # Write to sheet with optimized approach
            if len(data_to_write) > 1000:
                # For very large datasets, consider chunking (though unlikely in this use case)
                print(f"Large dataset detected ({len(data_to_write)} rows). Writing in optimized batch.")

            # Ensure the grid is big enough - reports are wider now that more products
            # are tracked, and the tabs were originally created with only 20 columns
            needed_cols = len(df_copy.columns)
            needed_rows = len(data_to_write) + 10
            try:
                if worksheet.col_count < needed_cols or worksheet.row_count < needed_rows:
                    self._api_call_with_retry(lambda: worksheet.resize(
                        rows=max(worksheet.row_count, needed_rows),
                        cols=max(worksheet.col_count, needed_cols)))
                    self._add_api_delay(0.3)
            except Exception as e:
                print(f"Warning: could not resize worksheet grid: {e}")

            self._api_call_with_retry(lambda: worksheet.update(values=data_to_write, range_name='A1'))
            self._add_api_delay(0.3)  # Delay after data write
            
            # Apply formatting
            self._apply_sheet_formatting(worksheet, df_copy, title)
            
            # Add dropdown validation
            self._add_dropdown_validation(worksheet, df_copy, title)

            # Trim leftover empty columns so the tab ends right at the data
            # (the grid may be wider from a previous run that had more products)
            try:
                final_cols = len(df_copy.columns)
                if worksheet.col_count > final_cols:
                    self._api_call_with_retry(lambda: worksheet.resize(
                        rows=worksheet.row_count, cols=final_cols))
            except Exception as e:
                print(f"Warning: could not trim worksheet columns: {e}")

            print(f"Successfully updated sheet: {sheet_name}")
            
        except Exception as e:
            print(f"Error updating sheet {sheet_name}: {e}")
    
    def _preserve_existing_data(self, worksheet, new_df, sheet_type):
        """Preserve existing comments and resolution data with optimized API calls"""
        try:
            # Skip preservation for pure-calculation reports (no manual comment/
            # resolution columns to keep): monthly summary, target tracker, performance
            if sheet_type in ('summary', 'target_tracker', 'performance'):
                return new_df
            
            # Read existing data using get_all_values with delay
            all_values = self._api_call_with_retry(lambda: worksheet.get_all_values())
            self._add_api_delay(0.1)  # Small delay after reading existing data
            
            if len(all_values) <= 3:  # No data rows
                return new_df
            
            # Skip title, timestamp and header rows, get data starting from row 5
            headers = all_values[3]  # Row 4 contains headers (0-indexed)
            data_rows = all_values[4:]  # Data starts from row 5
            
            if not data_rows:
                return new_df
            
            # Create DataFrame with headers, handling duplicates
            existing_df = pd.DataFrame(data_rows, columns=headers)
            
            # Create lookup for existing data
            for idx, new_row in new_df.iterrows():
                # Match on Date + Purchase Officer (Weight/Invoice discrepancy reports)
                if 'Date' in existing_df.columns and 'Date' in new_df.columns:
                    key_match = existing_df[
                        (existing_df['Date'] == new_row['Date']) &
                        (existing_df['Purchase Officer'] == new_row['Purchase Officer'])
                    ]
                else:
                    # Skip if no matching key columns found
                    continue
                
                if not key_match.empty:
                    existing_row = key_match.iloc[0]
                    
                    # Preserve existing response data only if they exist and are not default values
                    existing_responsible = existing_row.get('Responsible Party', '')
                    existing_root_cause = existing_row.get('Root Cause', '')
                    current_status = new_row['Status']
                    
                    # For matches, clear all preserved data since discrepancy is resolved in source data
                    if current_status in ['Match', 'MATCH']:
                        new_df.at[idx, 'Responsible Party'] = ''
                        new_df.at[idx, 'Root Cause'] = ''
                        new_df.at[idx, 'Resolution Status'] = 'RESOLVED'
                        new_df.at[idx, 'Resolution Date'] = ''
                        # Clear comments for resolved matches
                        if sheet_type == 'purchase':
                            new_df.at[idx, 'Purchase Team Comments'] = ''
                        else:  # inventory
                            new_df.at[idx, 'Inventory Team Comments'] = ''
                    else:
                        # For discrepancies, preserve existing data or leave empty
                        new_df.at[idx, 'Responsible Party'] = existing_responsible if existing_responsible and existing_responsible != 'N/A' else ''
                        new_df.at[idx, 'Root Cause'] = existing_root_cause if existing_root_cause and existing_root_cause != 'N/A' else ''
                        
                        # Smart status logic for discrepancies
                        existing_status = existing_row.get('Resolution Status', 'PENDING')
                        if existing_status == 'RESOLVED':
                            new_df.at[idx, 'Resolution Status'] = 'PENDING'  # Was resolved but issue returned
                        else:
                            new_df.at[idx, 'Resolution Status'] = existing_status
                        
                        # Keep Resolution Date empty for manual entry
                        new_df.at[idx, 'Resolution Date'] = ''
                        
                        # Preserve comments based on sheet type for ongoing discrepancies
                        if sheet_type == 'purchase':
                            new_df.at[idx, 'Purchase Team Comments'] = existing_row.get('Purchase Team Comments', '')
                        else:  # inventory
                            new_df.at[idx, 'Inventory Team Comments'] = existing_row.get('Inventory Team Comments', '')
                        
        except Exception as e:
            print(f"Error preserving existing data: {e}")
            
        return new_df
    
    def _customize_columns_for_sheet(self, df, sheet_type):
        """Customize columns based on which sheet (purchase vs inventory)"""
        if sheet_type == 'purchase':
            # Remove inventory team comments column if it exists
            if 'Inventory Team Comments' in df.columns:
                df = df.drop('Inventory Team Comments', axis=1)
        else:  # inventory
            # Remove purchase team comments column if it exists
            if 'Purchase Team Comments' in df.columns:
                df = df.drop('Purchase Team Comments', axis=1)
        
        return df
    
    def _apply_sheet_formatting(self, worksheet, df, title):
        """Apply beautiful formatting to the worksheet"""
        try:
            # Define colors
            colors = {
                'title': {'red': 0.18, 'green': 0.33, 'blue': 0.58},  # Dark blue
                'timestamp': {'red': 0.96, 'green': 0.96, 'blue': 0.96},  # Light gray
                'header': {'red': 0.91, 'green': 0.96, 'blue': 0.99},  # Light blue
                'match': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Light green
                'discrepancy': {'red': 1.0, 'green': 0.95, 'blue': 0.91},  # Light orange
                'missing': {'red': 1.0, 'green': 0.91, 'blue': 0.91},  # Light red
                'explainer_header': {'red': 0.95, 'green': 0.95, 'blue': 0.85},  # Light yellow/cream
                'explainer_text': {'red': 0.98, 'green': 0.98, 'blue': 0.95}  # Very light cream
            }

            # Get the worksheet ID
            worksheet_id = worksheet.id

            # Prepare batch update requests
            requests = []

            # Format title row (A1)
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': 1,
                        'startColumnIndex': 0,
                        'endColumnIndex': len(df.columns)
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': colors['title'],
                            'textFormat': {
                                'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                'fontSize': 14,
                                'bold': True
                            },
                            'horizontalAlignment': 'CENTER'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            })

            # Format timestamp row (A2)
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': 1,
                        'endRowIndex': 2,
                        'startColumnIndex': 0,
                        'endColumnIndex': len(df.columns)
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': colors['timestamp'],
                            'textFormat': {
                                'foregroundColor': {'red': 0.4, 'green': 0.4, 'blue': 0.4},
                                'fontSize': 10,
                                'bold': False,
                                'italic': True
                            },
                            'horizontalAlignment': 'RIGHT'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            })

            # Determine header row index based on report type
            is_performance = 'PERFORMANCE' in title
            is_target_tracker = 'PURCHASE TARGET' in title
            if is_performance:
                header_row_index = 12
            elif is_target_tracker:
                header_row_index = 11
            else:
                header_row_index = 3

            # Format explainer section for target tracker
            if is_target_tracker:
                self._format_explainer_block(
                    worksheet_id, requests, colors,
                    header_row_idx=3,
                    text_start_idx=4,
                    text_end_idx=10,
                    num_columns=len(df.columns),
                )

            # Format explainer section for performance reports
            elif is_performance:
                self._format_explainer_block(
                    worksheet_id, requests, colors,
                    header_row_idx=3,
                    text_start_idx=4,
                    text_end_idx=11,
                    num_columns=len(df.columns),
                )

            # Format header row
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': header_row_index,
                        'endRowIndex': header_row_index + 1,
                        'startColumnIndex': 0,
                        'endColumnIndex': len(df.columns)
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': colors['header'],
                            'textFormat': {
                                'fontSize': 11,
                                'bold': True
                            },
                            'horizontalAlignment': 'CENTER'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            })
            
            # Special formatting for Monthly Summary Report
            if 'MONTHLY SUMMARY' in title:
                self._apply_monthly_summary_formatting(worksheet_id, df, requests)
            
            # Special formatting for Performance Report
            elif 'PERFORMANCE' in title:
                self._apply_performance_formatting(worksheet_id, df, requests)

            # Special formatting for Purchase Target Tracker
            elif 'PURCHASE TARGET' in title:
                self._apply_target_tracker_formatting(worksheet_id, df, requests)

            # Format data rows based on Status and Resolution Status columns
            elif 'Status' in df.columns:
                status_col_index = df.columns.get_loc('Status')
                resolution_col_index = df.columns.get_loc('Resolution Status') if 'Resolution Status' in df.columns else None
                
                # All percentage-difference column indices (birds + every product)
                pct_col_indices = [i for i, col in enumerate(df.columns) if 'Percentage Difference' in col]
                
                for row_idx, row_data in enumerate(df.values, 4):  # Starting from row 5 (index 4)
                    status_value = str(row_data[status_col_index])
                    resolution_status = str(row_data[resolution_col_index]) if resolution_col_index is not None else 'PENDING'
                    
                    # Choose color based on resolution status first, then status
                    if resolution_status == 'RESOLVED':
                        bg_color = colors['match']  # Green for resolved
                    elif resolution_status == 'IN_PROGRESS':
                        bg_color = {'red': 1.0, 'green': 1.0, 'blue': 0.8}  # Light yellow for in progress
                    elif resolution_status == 'ESCALATED':
                        bg_color = {'red': 1.0, 'green': 0.8, 'blue': 0.8}  # Light red for escalated
                    elif status_value == 'Match' or status_value == 'MATCH':
                        bg_color = colors['match']
                    elif 'Discrepancy' in status_value or 'MISMATCH' in status_value:
                        bg_color = colors['discrepancy']
                    elif 'MISSING IN INVENTORY' in status_value:
                        bg_color = colors['missing']
                    elif 'MISSING IN PURCHASE' in status_value:
                        bg_color = colors['missing']
                    else:
                        continue
                    
                    # Apply formatting to the entire row
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': 0,
                                'endColumnIndex': len(df.columns)
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': bg_color,
                                    'textFormat': {
                                        'fontSize': 10
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                    
                    # Check for percentage discrepancies >0.6% or <-0.6% and highlight those cells
                    # Different colors for losses vs gains with enhanced contrast
                    loss_penalty_color = {'red': 1.0, 'green': 0.6, 'blue': 0.6}  # Light red for losses
                    gain_penalty_color = {'red': 1.0, 'green': 0.85, 'blue': 0.7}  # Medium peach for gains

                    for pct_col_idx in pct_col_indices:
                        if pct_col_idx is not None:
                            pct_value_str = str(row_data[pct_col_idx])
                            if pct_value_str != 'N/A' and pct_value_str.endswith('%'):
                                try:
                                    pct_value = float(pct_value_str.replace('%', ''))
                                    if pct_value < -0.6:  # Significant losses
                                        penalty_color = loss_penalty_color
                                    elif pct_value > 0.6:  # Significant gains
                                        penalty_color = gain_penalty_color
                                    else:
                                        penalty_color = None  # No penalty color needed

                                    if penalty_color is not None:
                                        # Highlight this specific cell with appropriate penalty color
                                        requests.append({
                                            'repeatCell': {
                                                'range': {
                                                    'sheetId': worksheet_id,
                                                    'startRowIndex': row_idx,
                                                    'endRowIndex': row_idx + 1,
                                                    'startColumnIndex': pct_col_idx,
                                                    'endColumnIndex': pct_col_idx + 1
                                                },
                                                'cell': {
                                                    'userEnteredFormat': {
                                                        'backgroundColor': penalty_color,
                                                        'textFormat': {
                                                            'fontSize': 10,
                                                            'bold': True
                                                        },
                                                        'horizontalAlignment': 'CENTER'
                                                    }
                                                },
                                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                                            }
                                        })
                                except ValueError:
                                    pass  # Skip if can't parse percentage
            
            # Auto-resize columns
            requests.append({
                'autoResizeDimensions': {
                    'dimensions': {
                        'sheetId': worksheet_id,
                        'dimension': 'COLUMNS',
                        'startIndex': 0,
                        'endIndex': len(df.columns)
                    }
                }
            })

            # Monthly Summary: wrap the long product headers and give the weight/birds
            # columns a comfortable uniform width (Month / Officer / Offtake stay auto)
            if 'MONTHLY SUMMARY' in title:
                keep_auto = {'Month', 'Purchase Officer', 'Offtake Count'}
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': header_row_index,
                            'endRowIndex': header_row_index + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {'wrapStrategy': 'WRAP', 'verticalAlignment': 'MIDDLE'}
                        },
                        'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                    }
                })
                for col_idx, col_name in enumerate(df.columns):
                    if col_name not in keep_auto:
                        requests.append({
                            'updateDimensionProperties': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'dimension': 'COLUMNS',
                                    'startIndex': col_idx,
                                    'endIndex': col_idx + 1
                                },
                                'properties': {'pixelSize': 165},
                                'fields': 'pixelSize'
                            }
                        })

            # Weight Discrepancy: same treatment - wrap the headers and give the
            # birds/weight/percentage columns a comfortable uniform width. The text
            # and workflow columns (dates, status, comments, resolution) stay auto.
            if 'DISCREPANCY' in title:
                keep_auto = {'Date', 'Purchase Officer', 'Status', 'Responsible Party',
                             'Root Cause', 'Purchase Team Comments', 'Inventory Team Comments',
                             'Resolution Status', 'Resolution Date'}
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': header_row_index,
                            'endRowIndex': header_row_index + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {'wrapStrategy': 'WRAP', 'verticalAlignment': 'MIDDLE'}
                        },
                        'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                    }
                })
                for col_idx, col_name in enumerate(df.columns):
                    if col_name not in keep_auto:
                        requests.append({
                            'updateDimensionProperties': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'dimension': 'COLUMNS',
                                    'startIndex': col_idx,
                                    'endIndex': col_idx + 1
                                },
                                'properties': {'pixelSize': 165},
                                'fields': 'pixelSize'
                            }
                        })

                # The free-text comment columns hold long sentences - cap their width
                # and wrap the text so a single long comment doesn't stretch the column
                comment_cols = ['Purchase Team Comments', 'Inventory Team Comments']
                data_start_row = header_row_index + 1
                data_end_row = data_start_row + len(df)
                for col_name in comment_cols:
                    if col_name in df.columns:
                        ci = df.columns.get_loc(col_name)
                        requests.append({
                            'updateDimensionProperties': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'dimension': 'COLUMNS',
                                    'startIndex': ci,
                                    'endIndex': ci + 1
                                },
                                'properties': {'pixelSize': 300},
                                'fields': 'pixelSize'
                            }
                        })
                        requests.append({
                            'repeatCell': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'startRowIndex': data_start_row,
                                    'endRowIndex': data_end_row,
                                    'startColumnIndex': ci,
                                    'endColumnIndex': ci + 1
                                },
                                'cell': {
                                    'userEnteredFormat': {'wrapStrategy': 'WRAP', 'verticalAlignment': 'TOP'}
                                },
                                'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                            }
                        })

            # Performance: same treatment - wrap the metric headers and give the
            # average/total/combined columns a comfortable uniform width
            # (Purchase Officer and Volume Category stay auto)
            if 'PERFORMANCE' in title:
                keep_auto = {'Purchase Officer', 'Volume Category'}
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': header_row_index,
                            'endRowIndex': header_row_index + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {'wrapStrategy': 'WRAP', 'verticalAlignment': 'MIDDLE'}
                        },
                        'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                    }
                })
                for col_idx, col_name in enumerate(df.columns):
                    if col_name not in keep_auto:
                        requests.append({
                            'updateDimensionProperties': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'dimension': 'COLUMNS',
                                    'startIndex': col_idx,
                                    'endIndex': col_idx + 1
                                },
                                'properties': {'pixelSize': 165},
                                'fields': 'pixelSize'
                            }
                        })

            # For Purchase Target Tracker, keep every column visible at first glance:
            # wrap the header row and cap data columns at a compact width so long
            # header names stack vertically instead of stretching the sheet.
            if is_target_tracker:
                # Wrap + center the header row (allows multi-line headers)
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': header_row_index,
                            'endRowIndex': header_row_index + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {
                                'wrapStrategy': 'WRAP',
                                'verticalAlignment': 'MIDDLE'
                            }
                        },
                        'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                    }
                })
                # Compact fixed width for data columns (Period stays auto-sized)
                period_idx = df.columns.get_loc('Period') if 'Period' in df.columns else 0
                for col_idx in range(len(df.columns)):
                    if col_idx == period_idx:
                        continue
                    requests.append({
                        'updateDimensionProperties': {
                            'range': {
                                'sheetId': worksheet_id,
                                'dimension': 'COLUMNS',
                                'startIndex': col_idx,
                                'endIndex': col_idx + 1
                            },
                            'properties': {'pixelSize': 95},
                            'fields': 'pixelSize'
                        }
                    })

            # For Invoice Mismatch Report, wrap every column that holds a long
            # comma-separated invoice list so nothing stretches unreadably wide
            if 'INVOICE MISMATCH' in title:
                # Wrap the header row so it reads consistently with the other tabs
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': header_row_index,
                            'endRowIndex': header_row_index + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {'wrapStrategy': 'WRAP', 'verticalAlignment': 'MIDDLE'}
                        },
                        'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                    }
                })
                long_text_cols = [
                    'Purchase Invoices', 'Inventory Invoices',
                    'Missing in Inventory', 'Extra in Inventory'
                ]
                invoice_col_indices = [
                    df.columns.get_loc(col)
                    for col in long_text_cols
                    if col in df.columns
                ]
                if invoice_col_indices:
                    data_start_row = header_row_index + 1
                    data_end_row = data_start_row + len(df)
                    for col_idx in invoice_col_indices:
                        # Override auto-resize with a sensible fixed width
                        requests.append({
                            'updateDimensionProperties': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'dimension': 'COLUMNS',
                                    'startIndex': col_idx,
                                    'endIndex': col_idx + 1
                                },
                                'properties': {'pixelSize': 280},
                                'fields': 'pixelSize'
                            }
                        })
                        # Wrap text and top-align in the invoice data cells
                        requests.append({
                            'repeatCell': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'startRowIndex': data_start_row,
                                    'endRowIndex': data_end_row,
                                    'startColumnIndex': col_idx,
                                    'endColumnIndex': col_idx + 1
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'wrapStrategy': 'WRAP',
                                        'verticalAlignment': 'TOP'
                                    }
                                },
                                'fields': 'userEnteredFormat(wrapStrategy,verticalAlignment)'
                            }
                        })

            # Force Lato across every textFormat so per-section requests that
            # replace the whole textFormat object don't drop the font.
            self._inject_font_family(requests, 'Lato')

            # Execute batch update
            if requests:
                self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                    'requests': requests
                }))
                self._add_api_delay(0.5)  # Small delay after formatting
                
        except Exception as e:
            print(f"Error applying formatting: {e}")

    def _inject_font_family(self, node, font):
        """Recursively set fontFamily on every textFormat dict in the requests.

        The per-section formatting requests use field masks like
        userEnteredFormat(...,textFormat,...) which replace the whole
        textFormat object. Adding fontFamily here keeps Lato on every cell
        across all output tabs.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'textFormat' and isinstance(value, dict):
                    value['fontFamily'] = font
                else:
                    self._inject_font_family(value, font)
        elif isinstance(node, list):
            for item in node:
                self._inject_font_family(item, font)

    def _format_explainer_block(self, worksheet_id, requests, colors,
                                header_row_idx, text_start_idx, text_end_idx,
                                num_columns):
        """Format an explainer section: merge each row across columns and wrap text.

        Merging lets long sentences wrap naturally across the full sheet width
        instead of overflowing into neighboring cells (which gets clipped once
        adjacent cells have their own formatting).
        """
        # Unfreeze any frozen columns so we can merge explainer rows across the
        # whole row (Google Sheets rejects merges that cross a freeze boundary).
        requests.append({
            'updateSheetProperties': {
                'properties': {
                    'sheetId': worksheet_id,
                    'gridProperties': {'frozenColumnCount': 0}
                },
                'fields': 'gridProperties.frozenColumnCount'
            }
        })

        # Unmerge first so re-runs don't collide with previously merged ranges.
        # Omit column bounds so this covers the FULL width of these rows - a prior
        # run may have merged across more columns (e.g. when more products had data),
        # and a narrower unmerge would only partially cover that merge, which the API
        # rejects ("must select all cells in a merged range") and aborts all formatting.
        requests.append({
            'unmergeCells': {
                'range': {
                    'sheetId': worksheet_id,
                    'startRowIndex': header_row_idx,
                    'endRowIndex': text_end_idx
                }
            }
        })

        # Style the header row
        requests.append({
            'repeatCell': {
                'range': {
                    'sheetId': worksheet_id,
                    'startRowIndex': header_row_idx,
                    'endRowIndex': header_row_idx + 1,
                    'startColumnIndex': 0,
                    'endColumnIndex': num_columns
                },
                'cell': {
                    'userEnteredFormat': {
                        'backgroundColor': colors['explainer_header'],
                        'textFormat': {'fontSize': 11, 'bold': True},
                        'horizontalAlignment': 'LEFT',
                        'verticalAlignment': 'MIDDLE',
                        'wrapStrategy': 'WRAP'
                    }
                },
                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)'
            }
        })

        # Style the text rows
        requests.append({
            'repeatCell': {
                'range': {
                    'sheetId': worksheet_id,
                    'startRowIndex': text_start_idx,
                    'endRowIndex': text_end_idx,
                    'startColumnIndex': 0,
                    'endColumnIndex': num_columns
                },
                'cell': {
                    'userEnteredFormat': {
                        'backgroundColor': colors['explainer_text'],
                        'textFormat': {'fontSize': 10, 'bold': False},
                        'horizontalAlignment': 'LEFT',
                        'verticalAlignment': 'MIDDLE',
                        'wrapStrategy': 'WRAP'
                    }
                },
                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)'
            }
        })

        # Merge each explainer row across all columns so wrap uses the full width
        for row_idx in range(header_row_idx, text_end_idx):
            requests.append({
                'mergeCells': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': row_idx,
                        'endRowIndex': row_idx + 1,
                        'startColumnIndex': 0,
                        'endColumnIndex': num_columns
                    },
                    'mergeType': 'MERGE_ALL'
                }
            })

    def _apply_monthly_summary_formatting(self, worksheet_id, df, requests):
        """Apply colorful formatting specific to monthly summary report"""
        try:
            # Enhanced color palette for monthly summary - light and easy on eyes
            summary_colors = {
                'offtake_count': {'red': 0.95, 'green': 0.90, 'blue': 0.98},  # Light lavender/purple
                'purchase_total': {'red': 0.91, 'green': 0.96, 'blue': 0.99},  # Very light blue (same as weight report)
                'inventory_total': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'difference_positive': {'red': 1.0, 'green': 0.95, 'blue': 0.91},  # Very light orange (same as weight report)
                'difference_negative': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'percentage_high': {'red': 1.0, 'green': 0.91, 'blue': 0.91},  # Very light red (same as weight report)
                'percentage_low': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'spacing_row': {'red': 1.0, 'green': 1.0, 'blue': 1.0}  # White for spacing
            }
            
            # Get column indices for different types
            offtake_col = df.columns.get_loc('Offtake Count') if 'Offtake Count' in df.columns else None
            purchase_cols = [i for i, col in enumerate(df.columns) if 'Purchase' in col and 'Total' in col]
            inventory_cols = [i for i, col in enumerate(df.columns) if 'Inventory' in col and 'Total' in col]
            difference_cols = [i for i, col in enumerate(df.columns) if 'Difference' in col and '%' not in col]
            percentage_cols = [i for i, col in enumerate(df.columns) if 'Percentage Difference' in col]
            
            for row_idx, row_data in enumerate(df.values, 4):  # Starting from row 5 (index 4)
                officer_name = str(row_data[df.columns.get_loc('Purchase Officer')])
                
                # Special formatting for analysis rows
                if 'GRAND TOTAL' in officer_name:
                    # Grand totals row
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': 0,
                                'endColumnIndex': len(df.columns)
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': {'red': 0.2, 'green': 0.3, 'blue': 0.6},  # Dark blue
                                    'textFormat': {
                                        'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                        'fontSize': 13,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                elif officer_name == '':
                    # Spacing row - keep white/transparent
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': 0,
                                'endColumnIndex': len(df.columns)
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': summary_colors['spacing_row']
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor)'
                        }
                    })
                else:
                    # Regular data rows with column-specific formatting
                    # Format offtake count column
                    if offtake_col is not None:
                        requests.append({
                            'repeatCell': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'startRowIndex': row_idx,
                                    'endRowIndex': row_idx + 1,
                                    'startColumnIndex': offtake_col,
                                    'endColumnIndex': offtake_col + 1
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'backgroundColor': summary_colors['offtake_count'],
                                        'textFormat': {'fontSize': 10, 'bold': True},
                                        'horizontalAlignment': 'CENTER'
                                    }
                                },
                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                            }
                        })

                    # Format purchase total columns
                    for col_idx in purchase_cols:
                        requests.append({
                            'repeatCell': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'startRowIndex': row_idx,
                                    'endRowIndex': row_idx + 1,
                                    'startColumnIndex': col_idx,
                                    'endColumnIndex': col_idx + 1
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'backgroundColor': summary_colors['purchase_total'],
                                        'textFormat': {'fontSize': 10, 'bold': True},
                                        'horizontalAlignment': 'CENTER'
                                    }
                                },
                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                            }
                        })
                    
                    # Format inventory total columns
                    for col_idx in inventory_cols:
                        requests.append({
                            'repeatCell': {
                                'range': {
                                    'sheetId': worksheet_id,
                                    'startRowIndex': row_idx,
                                    'endRowIndex': row_idx + 1,
                                    'startColumnIndex': col_idx,
                                    'endColumnIndex': col_idx + 1
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'backgroundColor': summary_colors['inventory_total'],
                                        'textFormat': {'fontSize': 10, 'bold': True},
                                        'horizontalAlignment': 'CENTER'
                                    }
                                },
                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                            }
                        })
                    
                    # Format difference columns based on positive/negative
                    for col_idx in difference_cols:
                        diff_value_str = str(row_data[col_idx])
                        if diff_value_str and diff_value_str != '':
                            try:
                                diff_value = float(diff_value_str.replace(',', ''))
                                bg_color = summary_colors['difference_positive'] if diff_value > 0 else summary_colors['difference_negative']
                                requests.append({
                                    'repeatCell': {
                                        'range': {
                                            'sheetId': worksheet_id,
                                            'startRowIndex': row_idx,
                                            'endRowIndex': row_idx + 1,
                                            'startColumnIndex': col_idx,
                                            'endColumnIndex': col_idx + 1
                                        },
                                        'cell': {
                                            'userEnteredFormat': {
                                                'backgroundColor': bg_color,
                                                'textFormat': {'fontSize': 10, 'bold': True},
                                                'horizontalAlignment': 'CENTER'
                                            }
                                        },
                                        'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                                    }
                                })
                            except ValueError:
                                pass
                    
                    # Format percentage columns based on threshold
                    for col_idx in percentage_cols:
                        pct_value_str = str(row_data[col_idx])
                        if pct_value_str and pct_value_str.endswith('%'):
                            try:
                                pct_value = float(pct_value_str.replace('%', ''))
                                if abs(pct_value) > 2.0:  # High discrepancy threshold
                                    bg_color = summary_colors['percentage_high']
                                    text_bold = True
                                else:
                                    bg_color = summary_colors['percentage_low']
                                    text_bold = False
                                
                                requests.append({
                                    'repeatCell': {
                                        'range': {
                                            'sheetId': worksheet_id,
                                            'startRowIndex': row_idx,
                                            'endRowIndex': row_idx + 1,
                                            'startColumnIndex': col_idx,
                                            'endColumnIndex': col_idx + 1
                                        },
                                        'cell': {
                                            'userEnteredFormat': {
                                                'backgroundColor': bg_color,
                                                'textFormat': {'fontSize': 10, 'bold': text_bold},
                                                'horizontalAlignment': 'CENTER'
                                            }
                                        },
                                        'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                                    }
                                })
                            except ValueError:
                                pass
                                
        except Exception as e:
            print(f"Error applying monthly summary formatting: {e}")
    
    def _apply_performance_formatting(self, worksheet_id, df, requests):
        """Apply colorful formatting specific to performance report"""
        try:
            # Unique warm color palette for performance report
            perf_colors = {
                'high_performer': {'red': 0.85, 'green': 0.95, 'blue': 0.85},    # Light mint green
                'good_performer': {'red': 0.90, 'green': 0.95, 'blue': 0.98},   # Very light cyan
                'average_performer': {'red': 0.95, 'green': 0.90, 'blue': 0.95}, # Light lavender
                'below_average': {'red': 0.98, 'green': 0.90, 'blue': 0.90},    # Light rose
                'officer_name': {'red': 0.75, 'green': 0.85, 'blue': 0.95},     # Soft blue
                'metrics': {'red': 0.95, 'green': 0.95, 'blue': 0.85},          # Light cream
                'working_days': {'red': 0.88, 'green': 0.92, 'blue': 0.88},     # Soft green
                'total_birds': {'red': 1.0, 'green': 0.90, 'blue': 0.85},       # Light coral/salmon
                'total_chicken': {'red': 0.89, 'green': 0.95, 'blue': 0.99},    # Light sky blue (all per-product totals)
                'combined_total': {'red': 0.95, 'green': 0.90, 'blue': 0.96}    # Light lavender/purple
            }

            # Get column indices
            officer_col = df.columns.get_loc('Purchase Officer') if 'Purchase Officer' in df.columns else None
            rating_col = df.columns.get_loc('Volume Category') if 'Volume Category' in df.columns else None
            days_col = df.columns.get_loc('Total Purchase Days') if 'Total Purchase Days' in df.columns else None

            # New total columns
            total_birds_col = df.columns.get_loc('Total Birds') if 'Total Birds' in df.columns else None
            # Every per-product total column (e.g. "Total Chicken Weight (kg)", "Total Head Weight (kg)")
            product_total_cols = [i for i, col in enumerate(df.columns)
                                  if col.startswith('Total ') and col.endswith('Weight (kg)')]
            combined_total_col = df.columns.get_loc('Combined Total (kg)') if 'Combined Total (kg)' in df.columns else None

            # Metric columns (averages)
            metric_cols = [i for i, col in enumerate(df.columns) if 'Average' in col]

            # Performance reports have explainer section, so data starts at row 14 (index 13)
            # Regular reports start at row 5 (index 4)
            data_start_row = 13

            for row_idx, row_data in enumerate(df.values, data_start_row):  # Starting from row 14 (index 13) for performance
                # Check if this is the grand total row
                officer_name = str(row_data[officer_col]) if officer_col is not None else ''
                is_grand_total = 'GRAND TOTAL' in officer_name

                if is_grand_total:
                    # Special formatting for grand total row
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': 0,
                                'endColumnIndex': len(df.columns)
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': {'red': 0.2, 'green': 0.3, 'blue': 0.6},  # Dark blue
                                    'textFormat': {
                                        'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                        'fontSize': 13,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                    continue  # Skip other formatting for grand total row

                # Get performance rating for color selection
                rating = str(row_data[rating_col]) if rating_col is not None else 'Average Performer'

                # Choose row color based on volume category
                if 'Highest Volume' in rating:
                    row_bg_color = perf_colors['high_performer']
                elif 'High Volume' in rating:
                    row_bg_color = perf_colors['good_performer']
                elif 'Lower Volume' in rating:
                    row_bg_color = perf_colors['below_average']
                else:
                    row_bg_color = perf_colors['average_performer']

                # Apply base formatting to entire row
                requests.append({
                    'repeatCell': {
                        'range': {
                            'sheetId': worksheet_id,
                            'startRowIndex': row_idx,
                            'endRowIndex': row_idx + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': len(df.columns)
                        },
                        'cell': {
                            'userEnteredFormat': {
                                'backgroundColor': row_bg_color,
                                'textFormat': {
                                    'fontSize': 10
                                },
                                'horizontalAlignment': 'CENTER'
                            }
                        },
                        'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                    }
                })
                
                # Special formatting for officer name column
                if officer_col is not None:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': officer_col,
                                'endColumnIndex': officer_col + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['officer_name'],
                                    'textFormat': {
                                        'fontSize': 11,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'LEFT'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                
                # Special formatting for metric columns with enhanced colors
                for col_idx in metric_cols:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': col_idx,
                                'endColumnIndex': col_idx + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['metrics'],
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                
                # Special formatting for working days column
                if days_col is not None:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': days_col,
                                'endColumnIndex': days_col + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['working_days'],
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': False
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                
                # Bold formatting for volume category column
                if rating_col is not None:
                    text_color = {'red': 0.2, 'green': 0.6, 'blue': 0.2} if 'Highest' in rating else {'red': 0.6, 'green': 0.2, 'blue': 0.2} if 'Lower' in rating else {'red': 0.3, 'green': 0.3, 'blue': 0.3}

                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': rating_col,
                                'endColumnIndex': rating_col + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': True,
                                        'foregroundColor': text_color
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(textFormat,horizontalAlignment)'
                        }
                    })

                # Special formatting for Total Birds column
                if total_birds_col is not None:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': total_birds_col,
                                'endColumnIndex': total_birds_col + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['total_birds'],
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })

                # Special formatting for every per-product Total Weight column
                for col_idx in product_total_cols:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': col_idx,
                                'endColumnIndex': col_idx + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['total_chicken'],
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })

                # Special formatting for Combined Total column
                if combined_total_col is not None:
                    requests.append({
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': row_idx,
                                'endRowIndex': row_idx + 1,
                                'startColumnIndex': combined_total_col,
                                'endColumnIndex': combined_total_col + 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': perf_colors['combined_total'],
                                    'textFormat': {
                                        'fontSize': 10,
                                        'bold': True
                                    },
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })

        except Exception as e:
            print(f"Error applying performance formatting: {e}")

    def _apply_target_tracker_formatting(self, worksheet_id, df, requests):
        """Apply formatting specific to the purchase target tracker report."""
        try:
            colors = {
                'monthly_bg': {'red': 0.96, 'green': 0.98, 'blue': 1.0},
                'quarterly_bg': {'red': 0.33, 'green': 0.53, 'blue': 0.80},
                'annual_bg': {'red': 0.18, 'green': 0.33, 'blue': 0.58},
                'white': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                'green_bg': {'red': 0.78, 'green': 0.94, 'blue': 0.78},
                'amber_bg': {'red': 1.0, 'green': 0.93, 'blue': 0.70},
                'red_bg': {'red': 1.0, 'green': 0.80, 'blue': 0.80},
                'deficit_red': {'red': 0.85, 'green': 0.20, 'blue': 0.20},
                'deficit_green': {'red': 0.13, 'green': 0.55, 'blue': 0.13},
                'adjusted_bg': {'red': 1.0, 'green': 0.95, 'blue': 0.80},
            }

            period_col = df.columns.get_loc('Period')
            achievement_col = df.columns.get_loc('Achievement %')
            deficit_cols = [df.columns.get_loc(c) for c in df.columns if 'Deficit' in c]
            adjusted_cols = [df.columns.get_loc(c) for c in df.columns if c.startswith('Adjusted')]
            actual_vs_adjusted_pairs = [
                (df.columns.get_loc('Actual Birds'), df.columns.get_loc('Adjusted Birds Target')),
                (df.columns.get_loc('Actual Vol (Kg)'), df.columns.get_loc('Adjusted Vol Target (Kg)')),
            ]
            num_cols = len(df.columns)

            # Data starts at row 13 (index 12) due to explainer section
            for row_idx, row_data in enumerate(df.values, 12):
                period = str(row_data[period_col])

                if period == '':
                    # Spacing row
                    requests.append({
                        'repeatCell': {
                            'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                      'startColumnIndex': 0, 'endColumnIndex': num_cols},
                            'cell': {'userEnteredFormat': {'backgroundColor': colors['white']}},
                            'fields': 'userEnteredFormat(backgroundColor)'
                        }
                    })
                elif 'ANNUAL' in period:
                    # Annual total row
                    requests.append({
                        'repeatCell': {
                            'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                      'startColumnIndex': 0, 'endColumnIndex': num_cols},
                            'cell': {'userEnteredFormat': {
                                'backgroundColor': colors['annual_bg'],
                                'textFormat': {'foregroundColor': colors['white'], 'fontSize': 13, 'bold': True},
                                'horizontalAlignment': 'CENTER'
                            }},
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                elif 'TOTAL' in period:
                    # Quarterly total row
                    requests.append({
                        'repeatCell': {
                            'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                      'startColumnIndex': 0, 'endColumnIndex': num_cols},
                            'cell': {'userEnteredFormat': {
                                'backgroundColor': colors['quarterly_bg'],
                                'textFormat': {'foregroundColor': colors['white'], 'fontSize': 11, 'bold': True},
                                'horizontalAlignment': 'CENTER'
                            }},
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })
                else:
                    # Monthly data row
                    requests.append({
                        'repeatCell': {
                            'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                      'startColumnIndex': 0, 'endColumnIndex': num_cols},
                            'cell': {'userEnteredFormat': {
                                'backgroundColor': colors['monthly_bg'],
                                'textFormat': {'fontSize': 10},
                                'horizontalAlignment': 'CENTER'
                            }},
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    })

                    # Achievement % coloring
                    ach_str = str(row_data[achievement_col]).replace('%', '')
                    try:
                        ach_val = float(ach_str)
                        if ach_val >= 100:
                            ach_bg = colors['green_bg']
                        elif ach_val >= 70:
                            ach_bg = colors['amber_bg']
                        else:
                            ach_bg = colors['red_bg']
                        requests.append({
                            'repeatCell': {
                                'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                          'startColumnIndex': achievement_col, 'endColumnIndex': achievement_col + 1},
                                'cell': {'userEnteredFormat': {
                                    'backgroundColor': ach_bg,
                                    'textFormat': {'fontSize': 10, 'bold': True},
                                    'horizontalAlignment': 'CENTER'
                                }},
                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                            }
                        })
                    except ValueError:
                        pass

                    # Deficit column text coloring
                    for col_idx in deficit_cols:
                        val_str = str(row_data[col_idx]).replace(',', '')
                        try:
                            val = float(val_str)
                            text_color = colors['deficit_red'] if val > 0 else colors['deficit_green']
                            requests.append({
                                'repeatCell': {
                                    'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                              'startColumnIndex': col_idx, 'endColumnIndex': col_idx + 1},
                                    'cell': {'userEnteredFormat': {
                                        'textFormat': {'foregroundColor': text_color, 'fontSize': 10, 'bold': True},
                                        'horizontalAlignment': 'CENTER'
                                    }},
                                    'fields': 'userEnteredFormat(textFormat,horizontalAlignment)'
                                }
                            })
                        except ValueError:
                            pass

                    # Tint Adjusted Target columns to highlight catch-up targets
                    for col_idx in adjusted_cols:
                        requests.append({
                            'repeatCell': {
                                'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                          'startColumnIndex': col_idx, 'endColumnIndex': col_idx + 1},
                                'cell': {'userEnteredFormat': {
                                    'backgroundColor': colors['adjusted_bg'],
                                    'textFormat': {'fontSize': 10, 'bold': True},
                                    'horizontalAlignment': 'CENTER'
                                }},
                                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                            }
                        })

                    # Color Actual cells based on whether they hit the Adjusted Target
                    for actual_idx, adjusted_idx in actual_vs_adjusted_pairs:
                        try:
                            actual_val = float(str(row_data[actual_idx]).replace(',', ''))
                            adjusted_val = float(str(row_data[adjusted_idx]).replace(',', ''))
                            text_color = colors['deficit_green'] if actual_val >= adjusted_val else colors['deficit_red']
                            requests.append({
                                'repeatCell': {
                                    'range': {'sheetId': worksheet_id, 'startRowIndex': row_idx, 'endRowIndex': row_idx + 1,
                                              'startColumnIndex': actual_idx, 'endColumnIndex': actual_idx + 1},
                                    'cell': {'userEnteredFormat': {
                                        'textFormat': {'foregroundColor': text_color, 'fontSize': 10, 'bold': True},
                                        'horizontalAlignment': 'CENTER'
                                    }},
                                    'fields': 'userEnteredFormat(textFormat,horizontalAlignment)'
                                }
                            })
                        except ValueError:
                            pass

        except Exception as e:
            print(f"Error applying target tracker formatting: {e}")

    def _add_dropdown_validation(self, worksheet, df, title):
        """Add dropdown validation to specific columns"""
        try:
            worksheet_id = worksheet.id

            # Clear ALL data validation from the entire sheet first
            # Use setDataValidation with no rule specified to reliably clear validation
            clear_all_validation_request = {
                'setDataValidation': {
                    'range': {
                        'sheetId': worksheet_id
                        # Not specifying startRowIndex/endRowIndex clears entire sheet
                    }
                    # No 'rule' parameter means clear all validation
                }
            }

            # Calculate dynamic row count for applying new validation
            # Use larger buffer for bigger datasets to ensure room for growth
            buffer_size = max(50, len(df) // 10)  # At least 50, or 10% of data size
            total_rows = len(df) + 4 + buffer_size  # Data rows + header rows + buffer
            max_rows = max(total_rows, 100)  # Ensure minimum of 100 rows for usability

            # Execute the clear all validation request first
            self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                'requests': [clear_all_validation_request]
            }))
            self._add_api_delay(0.5)  # Longer delay after clearing all validation

            # Define dropdown options (excluding Resolution Date)
            validation_rules = {
                'Responsible Party': ['Purchase Team', 'Inventory Team', 'Both Teams'],
                'Resolution Status': ['PENDING', 'RESOLVED', 'IN_PROGRESS', 'ESCALATED']
            }

            # Define root cause options based on report type
            if 'Weight' in title or 'BIRD COUNT' in title:
                validation_rules['Root Cause'] = [
                    'Scale Calibration Issue', 'Data Entry Error', 'Purchase Officer Mix-up',
                    'Inventory Officer Mix-up', 'Bird Count Error', 'Measurement Timing',
                    'Shrinkage/Spoilage', 'Recording Wrong Officer', 'Communication Gap'
                ]
            else:  # Invoice report
                validation_rules['Root Cause'] = [
                    'Data Entry Error', 'Missing Invoice', 'Communication Gap',
                    'Late Entry', 'Duplicate Entry', 'Invoice Number Typo', 'Missing Documentation',
                    'Recording Wrong Officer', 'Timing Issue'
                ]

            requests = []
            
            # Add validation for each column that exists in the dataframe
            for col_name, options in validation_rules.items():
                if col_name in df.columns:
                    col_index = df.columns.get_loc(col_name)
                    
                    # Create validation rule
                    validation_rule = {
                        'condition': {
                            'type': 'ONE_OF_LIST',
                            'values': [{'userEnteredValue': option} for option in options]
                        },
                        'showCustomUi': True,
                        'strict': False
                    }
                    
                    # Apply to data range (starting AFTER the header row)
                    # Row structure: 1=title, 2=timestamp, 3=empty, 4=headers, 5+=data
                    # In 0-indexed: 0=title, 1=timestamp, 2=empty, 3=headers, 4+=data
                    requests.append({
                        'setDataValidation': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': 4,  # Row 5 (1-based) - first data row, skipping header at row 3 (0-indexed)
                                'endRowIndex': max_rows,  # Dynamic based on actual data size
                                'startColumnIndex': col_index,
                                'endColumnIndex': col_index + 1
                            },
                            'rule': validation_rule
                        }
                    })
            
            # Execute validation requests
            if requests:
                self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                    'requests': requests
                }))
                self._add_api_delay(0.5)  # Small delay after validation
                
        except Exception as e:
            print(f"Error adding dropdown validation: {e}")
    
    def run_analysis(self, purchase_sheet_id, inventory_sheet_id):
        """Run the complete analysis"""
        print("Starting discrepancy analysis...")

        # Get configurable sheet names from environment variables
        purchase_sheet_name = os.getenv('PURCHASE_SHEET_NAME', 'Pullus Purchase Tracker')
        inventory_sheet_name = os.getenv('INVENTORY_SHEET_NAME', 'Pullus Inventory Tracker')

        # Read data from both sheets
        print(f"Reading purchase data from '{purchase_sheet_name}'...")
        purchase_df = self.read_sheet_data(purchase_sheet_id, purchase_sheet_name)

        print(f"Reading inventory data from '{inventory_sheet_name}'...")
        inventory_df = self.read_sheet_data(inventory_sheet_id, inventory_sheet_name)
        
        if purchase_df.empty or inventory_df.empty:
            print("Error: Could not read data from sheets")
            return None, None, None, None, None

        # Only report on products that actually have data (keeps empty columns out)
        self.active_products = self._compute_active_products(purchase_df, inventory_df)
        print(f"Products with data (shown in reports): {[p['label'] for p in self.active_products]}")

        # Process data
        print("Processing purchase data...")
        purchase_grouped = self.process_purchase_data(purchase_df)
        
        print("Processing inventory data...")
        inventory_processed = self.process_inventory_data(inventory_df)
        
        # Generate reports
        print("Generating weight discrepancy report...")
        weight_report = self.generate_weight_discrepancy_report(purchase_grouped, inventory_processed)
        
        print("Generating invoice mismatch report...")
        invoice_report = self.generate_invoice_mismatch_report(purchase_grouped, inventory_processed)
        
        print("Generating monthly summary report...")
        monthly_report = self.generate_monthly_summary_report(purchase_grouped, inventory_processed)
        
        print("Generating purchase officer performance report...")
        performance_report = self.generate_purchase_officer_performance_report(purchase_grouped)

        print("Generating purchase target tracker...")
        target_report = self.generate_purchase_target_report(purchase_grouped)

        # Update Google Sheets with optimized delays
        print("Updating purchase sheet with reports...")
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Weight Discrepancy Report", weight_report, 
                                "WEIGHT & BIRD COUNT DISCREPANCY ANALYSIS", "purchase")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates
        
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Invoice Mismatch Report", invoice_report, 
                                "INVOICE MISMATCH ANALYSIS", "purchase")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates
        
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Monthly Summary Report", monthly_report, 
                                "MONTHLY SUMMARY BY PURCHASE OFFICER", "summary")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates
        
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Purchase Officer Performance", performance_report,
                                "PURCHASE OFFICER PERFORMANCE ANALYSIS", "performance")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates

        if not target_report.empty:
            self.update_google_sheet_with_preservation(purchase_sheet_id, "Purchase Target Tracker", target_report,
                                    "2026 PURCHASE TARGET TRACKER", "target_tracker")
            self._add_api_delay(2.0)

        print("Updating inventory sheet with reports...")
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Weight Discrepancy Report", weight_report, 
                                "WEIGHT & BIRD COUNT DISCREPANCY ANALYSIS", "inventory")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates
        
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Invoice Mismatch Report", invoice_report, 
                                "INVOICE MISMATCH ANALYSIS", "inventory")
        self._add_api_delay(2.0)  # Longer delay between major sheet updates
        
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Monthly Summary Report", monthly_report, 
                                "MONTHLY SUMMARY BY PURCHASE OFFICER", "summary")
        
        print("Analysis complete!")
        return weight_report, invoice_report, monthly_report, performance_report, target_report

# Usage
if __name__ == "__main__":
    # Load .env file for local development (production environment variables take precedence)
    load_env_file()

    # Configuration from environment variables
    PURCHASE_SHEET_ID = os.getenv('PURCHASE_SHEET_ID')
    INVENTORY_SHEET_ID = os.getenv('INVENTORY_SHEET_ID')
    
    if not PURCHASE_SHEET_ID or not INVENTORY_SHEET_ID:
        raise ValueError("PURCHASE_SHEET_ID and INVENTORY_SHEET_ID environment variables must be set")
    
    # Initialize analyzer
    analyzer = DiscrepancyAnalyzer()
    
    # Run analysis
    weight_report, invoice_report, monthly_report, performance_report, target_report = analyzer.run_analysis(
        PURCHASE_SHEET_ID,
        INVENTORY_SHEET_ID
    )

    # Display summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Weight Discrepancies Found: {len(weight_report[weight_report['Status'] != 'Match'])}")
    print(f"Invoice Mismatches Found: {len(invoice_report[invoice_report['Status'] != 'MATCH'])}")
    print(f"Monthly Summary Records: {len(monthly_report)}")
    print(f"Purchase Officers Analyzed: {len(performance_report)}")
    print(f"Target Tracker Rows: {len(target_report)}")