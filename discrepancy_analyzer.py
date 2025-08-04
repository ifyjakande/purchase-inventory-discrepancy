import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import os
import json

class DiscrepancyAnalyzer:
    def __init__(self, service_account_json=None):
        """Initialize with service account credentials"""
        self.gc = self._authenticate(service_account_json)
        
    def _authenticate(self, service_account_json=None):
        """Authenticate with Google Sheets API"""
        scope = ['https://www.googleapis.com/auth/spreadsheets']
        
        if service_account_json:
            # Use provided JSON string (for GitHub Actions)
            service_account_info = json.loads(service_account_json)
            creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        else:
            # Fallback to environment variable or file
            service_account_json_env = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if service_account_json_env:
                service_account_info = json.loads(service_account_json_env)
                creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
            else:
                # Last fallback to local file (for development)
                service_account_file = "pullus-pipeline-40a5302e034d.json"
                creds = Credentials.from_service_account_file(service_account_file, scopes=scope)
        
        return gspread.authorize(creds)
    
    def read_sheet_data(self, sheet_id, sheet_name="Sheet1"):
        """Read data from Google Sheet"""
        try:
            sheet = self.gc.open_by_key(sheet_id)
            worksheet = sheet.worksheet(sheet_name)
            # Get all values and skip first 3 rows
            all_values = worksheet.get_all_values()
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
            print(f"Error reading sheet {sheet_id}: {e}")
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
        """Process inventory data"""
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
        
        return inventory_df
    
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
                bird_diff = round(purchase_row['NUMBER OF BIRDS'] - inv_row['NUMBER OF BIRDS'], 2)
                chicken_diff = round(purchase_row['PURCHASED CHICKEN WEIGHT'] - inv_row['INVENTORY CHICKEN WEIGHT'], 2)
                gizzard_diff = round(purchase_row['PURCHASED GIZZARD WEIGHT'] - inv_row['INVENTORY GIZZARD WEIGHT'], 2)
                
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
        
        return pd.DataFrame(discrepancies)
    
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
        
        return pd.DataFrame(mismatches)
    
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
            birds_diff = round(p_birds - i_birds, 2)
            chicken_diff = round(p_chicken - i_chicken, 2)
            gizzard_diff = round(p_gizzard - i_gizzard, 2)
            
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
    
    def _calculate_monthly_grand_total(self, month_df, month):
        """Calculate grand total for a specific month"""
        # Convert numeric strings back to numbers for totaling
        def parse_numeric(value_str):
            if isinstance(value_str, str) and value_str.replace(',', '').replace('.', '').replace('-', '').isdigit():
                return float(value_str.replace(',', ''))
            return 0
        
        # Calculate totals for each category for this month
        total_p_birds = sum(parse_numeric(str(val)) for val in month_df['Purchase Birds Total'])
        total_i_birds = sum(parse_numeric(str(val)) for val in month_df['Inventory Birds Total'])
        total_p_chicken = sum(parse_numeric(str(val)) for val in month_df['Purchase Chicken Weight Total'])
        total_i_chicken = sum(parse_numeric(str(val)) for val in month_df['Inventory Chicken Weight Total'])
        total_p_gizzard = sum(parse_numeric(str(val)) for val in month_df['Purchase Gizzard Weight Total'])
        total_i_gizzard = sum(parse_numeric(str(val)) for val in month_df['Inventory Gizzard Weight Total'])
        
        # Calculate total differences
        total_birds_diff = total_p_birds - total_i_birds
        total_chicken_diff = total_p_chicken - total_i_chicken
        total_gizzard_diff = total_p_gizzard - total_i_gizzard
        
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
        """Update Google Sheet with data preservation and customization"""
        try:
            sheet = self.gc.open_by_key(sheet_id)
            
            # Try to get existing worksheet, create if doesn't exist
            try:
                worksheet = sheet.worksheet(sheet_name)
            except:
                worksheet = sheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
            
            # Make a copy of the dataframe to avoid modifying original
            df_copy = df.copy()
            
            # Preserve existing data
            df_copy = self._preserve_existing_data(worksheet, df_copy, sheet_type)
            
            # Customize columns for the specific sheet
            df_copy = self._customize_columns_for_sheet(df_copy, sheet_type)
            
            # Clear the worksheet content and formatting
            worksheet.clear()
            
            # Clear all formatting by applying default formatting to the entire sheet
            try:
                worksheet.spreadsheet.batch_update({
                    'requests': [{
                        'repeatCell': {
                            'range': {
                                'sheetId': worksheet.id,
                                'startRowIndex': 0,
                                'endRowIndex': 1000,
                                'startColumnIndex': 0,
                                'endColumnIndex': 50
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
                })
            except Exception as e:
                print(f"Warning: Could not clear formatting: {e}")
            
            # Prepare data with title
            data_to_write = []
            data_to_write.append([title])  # Title row
            data_to_write.append([])  # Empty row
            data_to_write.append(df_copy.columns.tolist())  # Headers
            
            # Add data rows
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
            
            # Write to sheet
            worksheet.update(values=data_to_write, range_name='A1')
            
            # Apply formatting
            self._apply_sheet_formatting(worksheet, df_copy, title)
            
            # Add dropdown validation
            self._add_dropdown_validation(worksheet, df_copy, title)
            
            print(f"Successfully updated sheet: {sheet_name}")
            
        except Exception as e:
            print(f"Error updating sheet {sheet_name}: {e}")
    
    def _preserve_existing_data(self, worksheet, new_df, sheet_type):
        """Preserve existing comments and resolution data"""
        try:
            # Skip preservation for monthly summary reports - they're pure calculations
            if sheet_type == 'summary':
                return new_df
            # Read existing data using get_all_values to handle duplicate headers
            all_values = worksheet.get_all_values()
            if len(all_values) <= 3:  # No data rows
                return new_df
            
            # Skip title and header rows, get data starting from row 4
            headers = all_values[2]  # Row 3 contains headers (0-indexed)
            data_rows = all_values[3:]  # Data starts from row 4
            
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
            
            # Format header row (row 3)
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet_id,
                        'startRowIndex': 2,
                        'endRowIndex': 3,
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
            
            # Format data rows based on Status and Resolution Status columns
            elif 'Status' in df.columns:
                status_col_index = df.columns.get_loc('Status')
                resolution_col_index = df.columns.get_loc('Resolution Status') if 'Resolution Status' in df.columns else None
                
                # Get percentage column indices
                bird_pct_col = df.columns.get_loc('Birds Percentage Difference') if 'Birds Percentage Difference' in df.columns else None
                chicken_pct_col = df.columns.get_loc('Chicken Weight Percentage Difference') if 'Chicken Weight Percentage Difference' in df.columns else None
                gizzard_pct_col = df.columns.get_loc('Gizzard Weight Percentage Difference') if 'Gizzard Weight Percentage Difference' in df.columns else None
                
                for row_idx, row_data in enumerate(df.values, 3):  # Starting from row 4 (index 3)
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
                    penalty_color = {'red': 1.0, 'green': 0.6, 'blue': 0.6}  # Light red for penalties
                    
                    for pct_col_idx in [bird_pct_col, chicken_pct_col, gizzard_pct_col]:
                        if pct_col_idx is not None:
                            pct_value_str = str(row_data[pct_col_idx])
                            if pct_value_str != 'N/A' and pct_value_str.endswith('%'):
                                try:
                                    pct_value = float(pct_value_str.replace('%', ''))
                                    if pct_value > 1.0 or pct_value < -1.0:
                                        # Highlight this specific cell with penalty color
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
                worksheet.spreadsheet.batch_update({
                    'requests': requests
                })
                
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
            
            for row_idx, row_data in enumerate(df.values, 3):  # Starting from row 4 (index 3)
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
    
    def _add_dropdown_validation(self, worksheet, df, title):
        """Add dropdown validation to specific columns"""
        try:
            worksheet_id = worksheet.id
            
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
                    
                    # Apply to data range (starting from row 4, excluding headers)
                    requests.append({
                        'setDataValidation': {
                            'range': {
                                'sheetId': worksheet_id,
                                'startRowIndex': 3,  # Row 4 (0-indexed)
                                'endRowIndex': 1000,  # Up to row 1000
                                'startColumnIndex': col_index,
                                'endColumnIndex': col_index + 1
                            },
                            'rule': validation_rule
                        }
                    })
            
            # Execute validation requests
            if requests:
                worksheet.spreadsheet.batch_update({
                    'requests': requests
                })
                
        except Exception as e:
            print(f"Error adding dropdown validation: {e}")
    
    def run_analysis(self, purchase_sheet_id, inventory_sheet_id):
        """Run the complete analysis"""
        print("Starting discrepancy analysis...")
        
        # Read data from both sheets
        print("Reading purchase data...")
        purchase_df = self.read_sheet_data(purchase_sheet_id, "Pullus Purchase Tracker")
        
        print("Reading inventory data...")
        inventory_df = self.read_sheet_data(inventory_sheet_id, "Pullus Inventory Tracker")
        
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
        
        # Update Google Sheets
        print("Updating purchase sheet with reports...")
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Weight Discrepancy Report", weight_report, 
                                "WEIGHT & BIRD COUNT DISCREPANCY ANALYSIS", "purchase")
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Invoice Mismatch Report", invoice_report, 
                                "INVOICE MISMATCH ANALYSIS", "purchase")
        self.update_google_sheet_with_preservation(purchase_sheet_id, "Monthly Summary Report", monthly_report, 
                                "MONTHLY SUMMARY BY PURCHASE OFFICER", "summary")
        
        print("Updating inventory sheet with reports...")
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Weight Discrepancy Report", weight_report, 
                                "WEIGHT & BIRD COUNT DISCREPANCY ANALYSIS", "inventory")
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Invoice Mismatch Report", invoice_report, 
                                "INVOICE MISMATCH ANALYSIS", "inventory")
        self.update_google_sheet_with_preservation(inventory_sheet_id, "Monthly Summary Report", monthly_report, 
                                "MONTHLY SUMMARY BY PURCHASE OFFICER", "summary")
        
        print("Analysis complete!")
        return weight_report, invoice_report, monthly_report

# Usage
if __name__ == "__main__":
    # Configuration from environment variables
    PURCHASE_SHEET_ID = os.getenv('PURCHASE_SHEET_ID')
    INVENTORY_SHEET_ID = os.getenv('INVENTORY_SHEET_ID')
    
    if not PURCHASE_SHEET_ID or not INVENTORY_SHEET_ID:
        raise ValueError("PURCHASE_SHEET_ID and INVENTORY_SHEET_ID environment variables must be set")
    
    # Initialize analyzer
    analyzer = DiscrepancyAnalyzer()
    
    # Run analysis
    weight_report, invoice_report, monthly_report = analyzer.run_analysis(
        PURCHASE_SHEET_ID, 
        INVENTORY_SHEET_ID
    )
    
    # Display summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Weight Discrepancies Found: {len(weight_report[weight_report['Status'] != 'Match'])}")
    print(f"Invoice Mismatches Found: {len(invoice_report[invoice_report['Status'] != 'MATCH'])}")
    print(f"Monthly Summary Records: {len(monthly_report)}")