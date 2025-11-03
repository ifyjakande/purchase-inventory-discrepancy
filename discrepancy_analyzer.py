import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import os
import json
import time
import random
from googleapiclient.errors import HttpError
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

class DiscrepancyAnalyzer:
    def __init__(self, service_account_json=None):
        """Initialize with service account credentials"""
        self.gc = self._authenticate(service_account_json)

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
                numeric_columns = ['NUMBER OF BIRDS', 'PURCHASED CHICKEN WEIGHT', 'PURCHASED GIZZARD WEIGHT', 
                                 'INVENTORY CHICKEN WEIGHT', 'INVENTORY GIZZARD WEIGHT']
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
        grouped = purchase_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({
            'NUMBER OF BIRDS': 'sum',
            'PURCHASED CHICKEN WEIGHT': 'sum',
            'PURCHASED GIZZARD WEIGHT': 'sum',
            'INVOICE NUMBER': lambda x: list(x)  # Collect all invoice numbers
        }).reset_index()
        
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
        grouped = inventory_df.groupby(['DATE', 'PURCHASE OFFICER NAME']).agg({
            'NUMBER OF BIRDS': 'sum',
            'INVENTORY CHICKEN WEIGHT': 'sum',
            'INVENTORY GIZZARD WEIGHT': 'sum',
            'INVOICE_LIST': lambda x: [item for sublist in x for item in sublist]  # Flatten all invoice lists
        }).reset_index()
        
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
                
                # Calculate discrepancies (round to 2 decimal places)
                bird_diff = round(inv_row['NUMBER OF BIRDS'] - purchase_row['NUMBER OF BIRDS'], 2)
                chicken_diff = round(inv_row['INVENTORY CHICKEN WEIGHT'] - purchase_row['PURCHASED CHICKEN WEIGHT'], 2)
                gizzard_diff = round(inv_row['INVENTORY GIZZARD WEIGHT'] - purchase_row['PURCHASED GIZZARD WEIGHT'], 2)
                
                # Calculate percentage differences using purchase as baseline
                bird_pct_diff = round((bird_diff / purchase_row['NUMBER OF BIRDS']) * 100, 2) if purchase_row['NUMBER OF BIRDS'] > 0 else 0
                chicken_pct_diff = round((chicken_diff / purchase_row['PURCHASED CHICKEN WEIGHT']) * 100, 2) if purchase_row['PURCHASED CHICKEN WEIGHT'] > 0 else 0
                gizzard_pct_diff = round((gizzard_diff / purchase_row['PURCHASED GIZZARD WEIGHT']) * 100, 2) if purchase_row['PURCHASED GIZZARD WEIGHT'] > 0 else 0
                
                # Check if there's a discrepancy (use tolerance for floating point comparison)
                tolerance = 0.01  # 0.01 tolerance for floating point precision
                has_discrepancy = (abs(bird_diff) > tolerance or abs(chicken_diff) > tolerance or abs(gizzard_diff) > tolerance)
                
                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': f"{round(purchase_row['NUMBER OF BIRDS'], 2):,}",
                    'Inventory Birds': f"{round(inv_row['NUMBER OF BIRDS'], 2):,}",
                    'Birds Difference': f"{bird_diff:,}",
                    'Purchase Chicken Weight': f"{round(purchase_row['PURCHASED CHICKEN WEIGHT'], 2):,}",
                    'Inventory Chicken Weight': f"{round(inv_row['INVENTORY CHICKEN WEIGHT'], 2):,}",
                    'Chicken Weight Difference': f"{chicken_diff:,}",
                    'Purchase Gizzard Weight': f"{round(purchase_row['PURCHASED GIZZARD WEIGHT'], 2):,}",
                    'Inventory Gizzard Weight': f"{round(inv_row['INVENTORY GIZZARD WEIGHT'], 2):,}",
                    'Gizzard Weight Difference': f"{gizzard_diff:,}",
                    'Birds Percentage Difference': f"{bird_pct_diff}%",
                    'Chicken Weight Percentage Difference': f"{chicken_pct_diff}%",
                    'Gizzard Weight Percentage Difference': f"{gizzard_pct_diff}%",
                    'Status': 'Discrepancy' if has_discrepancy else 'Match',
                    'Responsible Party': '' if has_discrepancy else '',
                    'Root Cause': '' if has_discrepancy else '',
                    'Purchase Team Comments': '',
                    'Inventory Team Comments': '',
                    'Resolution Status': 'PENDING' if has_discrepancy else 'RESOLVED',
                    'Resolution Date': ''
                })
            else:
                # No inventory record found - purchase exists but not in inventory
                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': f"{round(purchase_row['NUMBER OF BIRDS'], 2):,}",
                    'Inventory Birds': 'NOT FOUND',
                    'Birds Difference': 'N/A',
                    'Purchase Chicken Weight': f"{round(purchase_row['PURCHASED CHICKEN WEIGHT'], 2):,}",
                    'Inventory Chicken Weight': 'NOT FOUND',
                    'Chicken Weight Difference': 'N/A',
                    'Purchase Gizzard Weight': f"{round(purchase_row['PURCHASED GIZZARD WEIGHT'], 2):,}",
                    'Inventory Gizzard Weight': 'NOT FOUND',
                    'Gizzard Weight Difference': 'N/A',
                    'Birds Percentage Difference': 'N/A',
                    'Chicken Weight Percentage Difference': 'N/A',
                    'Gizzard Weight Percentage Difference': 'N/A',
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
                discrepancies.append({
                    'Date': date.strftime('%d-%b-%Y'),
                    'Purchase Officer': officer,
                    'Purchase Birds': 'NOT FOUND',
                    'Inventory Birds': f"{round(inv_row['NUMBER OF BIRDS'], 2):,}",
                    'Birds Difference': 'N/A',
                    'Purchase Chicken Weight': 'NOT FOUND',
                    'Inventory Chicken Weight': f"{round(inv_row['INVENTORY CHICKEN WEIGHT'], 2):,}",
                    'Chicken Weight Difference': 'N/A',
                    'Purchase Gizzard Weight': 'NOT FOUND',
                    'Inventory Gizzard Weight': f"{round(inv_row['INVENTORY GIZZARD WEIGHT'], 2):,}",
                    'Gizzard Weight Difference': 'N/A',
                    'Birds Percentage Difference': 'N/A',
                    'Chicken Weight Percentage Difference': 'N/A',
                    'Gizzard Weight Percentage Difference': 'N/A',
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
            purchase_monthly = purchase_grouped.groupby(['YEAR_MONTH', 'PURCHASE OFFICER NAME']).agg({
                'NUMBER OF BIRDS': 'sum',
                'PURCHASED CHICKEN WEIGHT': 'sum',
                'PURCHASED GIZZARD WEIGHT': 'sum'
            }).reset_index()
        else:
            purchase_monthly = pd.DataFrame()
        
        # Process inventory data by month and officer
        if not inventory_df.empty:
            inventory_df['YEAR_MONTH'] = inventory_df['DATE'].dt.to_period('M')
            inventory_monthly = inventory_df.groupby(['YEAR_MONTH', 'PURCHASE OFFICER NAME']).agg({
                'NUMBER OF BIRDS': 'sum',
                'INVENTORY CHICKEN WEIGHT': 'sum',
                'INVENTORY GIZZARD WEIGHT': 'sum'
            }).reset_index()
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
            
            # Extract values or set to 0 if not found
            if not purchase_match.empty:
                p_birds = round(purchase_match.iloc[0]['NUMBER OF BIRDS'], 2)
                p_chicken = round(purchase_match.iloc[0]['PURCHASED CHICKEN WEIGHT'], 2)
                p_gizzard = round(purchase_match.iloc[0]['PURCHASED GIZZARD WEIGHT'], 2)
            else:
                p_birds = 0
                p_chicken = 0
                p_gizzard = 0
            
            if not inventory_match.empty:
                i_birds = round(inventory_match.iloc[0]['NUMBER OF BIRDS'], 2)
                i_chicken = round(inventory_match.iloc[0]['INVENTORY CHICKEN WEIGHT'], 2)
                i_gizzard = round(inventory_match.iloc[0]['INVENTORY GIZZARD WEIGHT'], 2)
            else:
                i_birds = 0
                i_chicken = 0
                i_gizzard = 0
            
            # Calculate differences
            birds_diff = round(i_birds - p_birds, 2)
            chicken_diff = round(i_chicken - p_chicken, 2)
            gizzard_diff = round(i_gizzard - p_gizzard, 2)
            
            # Calculate percentage differences
            birds_pct = round((birds_diff / p_birds) * 100, 2) if p_birds > 0 else 0
            chicken_pct = round((chicken_diff / p_chicken) * 100, 2) if p_chicken > 0 else 0
            gizzard_pct = round((gizzard_diff / p_gizzard) * 100, 2) if p_gizzard > 0 else 0
            
            summaries.append({
                'Month': str(year_month),
                'Purchase Officer': officer,
                'Purchase Birds Total': f"{p_birds:,.0f}",
                'Inventory Birds Total': f"{i_birds:,.0f}",
                'Birds Difference': f"{birds_diff:,.0f}",
                'Birds Percentage Difference': f"{birds_pct}%",
                'Purchase Chicken Weight Total': f"{p_chicken:,.2f}",
                'Inventory Chicken Weight Total': f"{i_chicken:,.2f}",
                'Chicken Weight Difference': f"{chicken_diff:,.2f}",
                'Chicken Weight Percentage Difference': f"{chicken_pct}%",
                'Purchase Gizzard Weight Total': f"{p_gizzard:,.2f}",
                'Inventory Gizzard Weight Total': f"{i_gizzard:,.2f}",
                'Gizzard Weight Difference': f"{gizzard_diff:,.2f}",
                'Gizzard Weight Percentage Difference': f"{gizzard_pct}%"
            })
        
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
    
    def generate_purchase_officer_performance_report(self, purchase_grouped):
        """Generate purchase officer performance report with averages"""
        if purchase_grouped.empty:
            return pd.DataFrame()
        
        # Calculate averages per purchase officer
        performance_stats = purchase_grouped.groupby('PURCHASE OFFICER NAME').agg({
            'NUMBER OF BIRDS': ['mean', 'count'],
            'PURCHASED CHICKEN WEIGHT': 'mean',
            'PURCHASED GIZZARD WEIGHT': 'mean'
        }).reset_index()
        
        # Flatten column names
        performance_stats.columns = [
            'Purchase Officer',
            'Average Birds per Day',
            'Total Purchase Days',
            'Average Chicken Weight per Day',
            'Average Gizzard Weight per Day'
        ]
        
        # Round to appropriate decimal places
        performance_stats['Average Birds per Day'] = performance_stats['Average Birds per Day'].round(0)
        performance_stats['Average Chicken Weight per Day'] = performance_stats['Average Chicken Weight per Day'].round(2)
        performance_stats['Average Gizzard Weight per Day'] = performance_stats['Average Gizzard Weight per Day'].round(2)
        
        # Calculate data-driven performance thresholds
        bird_averages = performance_stats['Average Birds per Day'].values
        q75 = float(performance_stats['Average Birds per Day'].quantile(0.75))
        q50 = float(performance_stats['Average Birds per Day'].quantile(0.50))
        q25 = float(performance_stats['Average Birds per Day'].quantile(0.25))
        
        # Format for display
        performance_report = []
        for _, row in performance_stats.iterrows():
            performance_report.append({
                'Purchase Officer': row['Purchase Officer'],
                'Average Birds per Day': f"{row['Average Birds per Day']:,.0f}",
                'Average Chicken Weight per Day (kg)': f"{row['Average Chicken Weight per Day']:,.2f}",
                'Average Gizzard Weight per Day (kg)': f"{row['Average Gizzard Weight per Day']:,.2f}",
                'Total Purchase Days': f"{int(row['Total Purchase Days']):,}",
                'Volume Category': self._calculate_data_driven_performance_rating(
                    row['Average Birds per Day'], q75, q50, q25
                )
            })
        
        # Sort by average birds per day (descending)
        performance_df = pd.DataFrame(performance_report)
        performance_df = performance_df.sort_values('Average Birds per Day', 
                                                   key=lambda x: x.str.replace(',', '').astype(float), 
                                                   ascending=False).reset_index(drop=True)
        
        return performance_df
    
    def _calculate_data_driven_performance_rating(self, avg_birds, q75, q50, q25):
        """Calculate volume category based on data distribution percentiles"""
        if avg_birds >= q75:
            return "Highest Volume"      # Top 25%
        elif avg_birds >= q50:
            return "High Volume"         # Above median (50th-75th percentile)
        elif avg_birds >= q25:
            return "Moderate Volume"     # Below median but above bottom 25%
        else:
            return "Lower Volume"        # Bottom 25%

    def _calculate_monthly_grand_total(self, month_df, month):
        """Calculate grand total for a specific month"""
        # Convert numeric strings back to numbers for totaling
        def parse_numeric(value_str):
            try:
                return float(str(value_str).replace(',', ''))
            except (ValueError, AttributeError):
                return 0
        
        # Calculate totals for each category for this month
        total_p_birds = sum(parse_numeric(str(val)) for val in month_df['Purchase Birds Total'])
        total_i_birds = sum(parse_numeric(str(val)) for val in month_df['Inventory Birds Total'])
        total_p_chicken = sum(parse_numeric(str(val)) for val in month_df['Purchase Chicken Weight Total'])
        total_i_chicken = sum(parse_numeric(str(val)) for val in month_df['Inventory Chicken Weight Total'])
        total_p_gizzard = sum(parse_numeric(str(val)) for val in month_df['Purchase Gizzard Weight Total'])
        total_i_gizzard = sum(parse_numeric(str(val)) for val in month_df['Inventory Gizzard Weight Total'])
        
        # Calculate total differences
        total_birds_diff = total_i_birds - total_p_birds
        total_chicken_diff = total_i_chicken - total_p_chicken
        total_gizzard_diff = total_i_gizzard - total_p_gizzard
        
        # Calculate total percentage differences
        total_birds_pct = round((total_birds_diff / total_p_birds) * 100, 2) if total_p_birds > 0 else 0
        total_chicken_pct = round((total_chicken_diff / total_p_chicken) * 100, 2) if total_p_chicken > 0 else 0
        total_gizzard_pct = round((total_gizzard_diff / total_p_gizzard) * 100, 2) if total_p_gizzard > 0 else 0
        
        # Helper function to format weight with appropriate unit
        def format_weight(weight):
            if abs(weight) >= 1000:
                tonnes_value = weight / 1000
                unit = "tonne" if abs(tonnes_value) == 1.00 else "tonnes"
                return f"{tonnes_value:,.2f} {unit}"
            else:
                return f"{weight:,.2f} kg"
        
        return {
            'Month': '',
            'Purchase Officer': f'═══════ {month} GRAND TOTAL ═══════',
            'Purchase Birds Total': f"{total_p_birds:,.0f}",
            'Inventory Birds Total': f"{total_i_birds:,.0f}",
            'Birds Difference': f"{total_birds_diff:,.0f}",
            'Birds Percentage Difference': f"{total_birds_pct}%",
            'Purchase Chicken Weight Total': format_weight(total_p_chicken),
            'Inventory Chicken Weight Total': format_weight(total_i_chicken),
            'Chicken Weight Difference': format_weight(total_chicken_diff),
            'Chicken Weight Percentage Difference': f"{total_chicken_pct}%",
            'Purchase Gizzard Weight Total': format_weight(total_p_gizzard),
            'Inventory Gizzard Weight Total': format_weight(total_i_gizzard),
            'Gizzard Weight Difference': format_weight(total_gizzard_diff),
            'Gizzard Weight Percentage Difference': f"{total_gizzard_pct}%"
        }
    
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
            
            # Clear all formatting with optimized batch size
            try:
                # Smaller batch size to prevent timeouts
                self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                    'requests': [{
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet.id,
                                'startRowIndex': 0,
                                'endRowIndex': len(df_copy) + 10,  # Scale with actual data size
                                'startColumnIndex': 0,
                                'endColumnIndex': len(df_copy.columns) + 5  # Scale with actual column count
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                                    'textFormat': {
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
            
            self._api_call_with_retry(lambda: worksheet.update(values=data_to_write, range_name='A1'))
            self._add_api_delay(0.3)  # Delay after data write
            
            # Apply formatting
            self._apply_sheet_formatting(worksheet, df_copy, title)
            
            # Add dropdown validation
            self._add_dropdown_validation(worksheet, df_copy, title)
            
            print(f"Successfully updated sheet: {sheet_name}")
            
        except Exception as e:
            print(f"Error updating sheet {sheet_name}: {e}")
    
    def _preserve_existing_data(self, worksheet, new_df, sheet_type):
        """Preserve existing comments and resolution data with optimized API calls"""
        try:
            # Skip preservation for monthly summary reports - they're pure calculations
            if sheet_type == 'summary':
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
                # Create unique key for matching based on available columns
                if 'Date' in existing_df.columns and 'Date' in new_df.columns:
                    # For discrepancy reports (Weight/Invoice)
                    key_match = existing_df[
                        (existing_df['Date'] == new_row['Date']) & 
                        (existing_df['Purchase Officer'] == new_row['Purchase Officer'])
                    ]
                elif 'Month' in existing_df.columns and 'Month' in new_df.columns:
                    # For monthly summary reports
                    key_match = existing_df[
                        (existing_df['Month'] == new_row['Month']) & 
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
                'missing': {'red': 1.0, 'green': 0.91, 'blue': 0.91}  # Light red
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
            
            # Format header row (row 4)
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': 3,
                        'endRowIndex': 4,
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
            
            # Format data rows based on Status and Resolution Status columns
            elif 'Status' in df.columns:
                status_col_index = df.columns.get_loc('Status')
                resolution_col_index = df.columns.get_loc('Resolution Status') if 'Resolution Status' in df.columns else None
                
                # Get percentage column indices
                bird_pct_col = df.columns.get_loc('Birds Percentage Difference') if 'Birds Percentage Difference' in df.columns else None
                chicken_pct_col = df.columns.get_loc('Chicken Weight Percentage Difference') if 'Chicken Weight Percentage Difference' in df.columns else None
                gizzard_pct_col = df.columns.get_loc('Gizzard Weight Percentage Difference') if 'Gizzard Weight Percentage Difference' in df.columns else None
                
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
                    
                    # Check for percentage discrepancies >1% or <-1% and highlight those cells
                    # Different colors for losses vs gains with enhanced contrast
                    loss_penalty_color = {'red': 1.0, 'green': 0.8, 'blue': 0.8}  # Medium rose for losses
                    gain_penalty_color = {'red': 1.0, 'green': 0.85, 'blue': 0.7}  # Medium peach for gains

                    for pct_col_idx in [bird_pct_col, chicken_pct_col, gizzard_pct_col]:
                        if pct_col_idx is not None:
                            pct_value_str = str(row_data[pct_col_idx])
                            if pct_value_str != 'N/A' and pct_value_str.endswith('%'):
                                try:
                                    pct_value = float(pct_value_str.replace('%', ''))
                                    if pct_value < -1.0:  # Significant losses
                                        penalty_color = loss_penalty_color
                                    elif pct_value > 1.0:  # Significant gains
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
            
            # Execute batch update
            if requests:
                self._api_call_with_retry(lambda: worksheet.spreadsheet.batch_update({
                    'requests': requests
                }))
                self._add_api_delay(0.5)  # Small delay after formatting
                
        except Exception as e:
            print(f"Error applying formatting: {e}")
    
    def _apply_monthly_summary_formatting(self, worksheet_id, df, requests):
        """Apply colorful formatting specific to monthly summary report"""
        try:
            # Enhanced color palette for monthly summary - light and easy on eyes
            summary_colors = {
                'purchase_total': {'red': 0.91, 'green': 0.96, 'blue': 0.99},  # Very light blue (same as weight report)
                'inventory_total': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'difference_positive': {'red': 1.0, 'green': 0.95, 'blue': 0.91},  # Very light orange (same as weight report)
                'difference_negative': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'percentage_high': {'red': 1.0, 'green': 0.91, 'blue': 0.91},  # Very light red (same as weight report)
                'percentage_low': {'red': 0.91, 'green': 0.96, 'blue': 0.91},  # Very light green (same as weight report)
                'spacing_row': {'red': 1.0, 'green': 1.0, 'blue': 1.0}  # White for spacing
            }
            
            # Get column indices for different types
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
                'working_days': {'red': 0.88, 'green': 0.92, 'blue': 0.88}      # Soft green
            }
            
            # Get column indices
            officer_col = df.columns.get_loc('Purchase Officer') if 'Purchase Officer' in df.columns else None
            rating_col = df.columns.get_loc('Volume Category') if 'Volume Category' in df.columns else None
            days_col = df.columns.get_loc('Total Purchase Days') if 'Total Purchase Days' in df.columns else None
            
            # Metric columns (averages)
            metric_cols = [i for i, col in enumerate(df.columns) if 'Average' in col]
            
            for row_idx, row_data in enumerate(df.values, 4):  # Starting from row 5 (index 4)
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
                                
        except Exception as e:
            print(f"Error applying performance formatting: {e}")
    
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
            return None, None
        
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
        return weight_report, invoice_report, monthly_report, performance_report

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
    weight_report, invoice_report, monthly_report, performance_report = analyzer.run_analysis(
        PURCHASE_SHEET_ID, 
        INVENTORY_SHEET_ID
    )
    
    # Display summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Weight Discrepancies Found: {len(weight_report[weight_report['Status'] != 'Match'])}")
    print(f"Invoice Mismatches Found: {len(invoice_report[invoice_report['Status'] != 'MATCH'])}")
    print(f"Monthly Summary Records: {len(monthly_report)}")
    print(f"Purchase Officers Analyzed: {len(performance_report)}")