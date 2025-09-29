import requests
import pandas as pd
from io import StringIO
import re

def create_postman_request(url: str, query: str) -> requests.Response:
    try:
        from .config import get_database_headers
        headers = get_database_headers()
    except ImportError:
        # Fallback for direct execution
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from config import get_database_headers
            headers = get_database_headers()
        except ImportError:
            # Fallback to basic headers without authentication
            headers = {
                'Content-Type': 'text/plain',
                'Accept': 'application/json',
            }
    
    # Add timeout and chunked transfer handling
    try:
        # Use a longer timeout for large queries
        response = requests.post(url, headers=headers, data=query, timeout=60, stream=True)
        
        # Handle chunked transfer to prevent IncompleteRead errors
        if response.status_code == 200:
            # Read the response in chunks to handle large datasets
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
            response._content = content
        
        return response
    except requests.exceptions.Timeout:
        print("⚠️ Database request timed out. Consider reducing date range or adding filters.")
        raise
    except requests.exceptions.ConnectionError as e:
        print(f"⚠️ Database connection error: {e}")
        raise
    except Exception as e:
        print(f"⚠️ Unexpected error in database request: {e}")
        raise

def clean_column_values(df, column_name):
    """Clean column values by removing (tbr), (na), (v) suffixes and trimming whitespace"""
    df[column_name] = df[column_name].astype(str).str.replace(r'\(tbr\)|\(na\)|\(v\)', '', flags=re.IGNORECASE, regex=True).str.strip()
    return df

def get_spot_data(products: str, start_date: str, end_date: str, regions: list = None) -> pd.DataFrame:
    import time
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            from .config import get_database_url
            url = get_database_url()
        except ImportError:
            # Fallback for direct execution
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            try:
                from config import get_database_url
                url = get_database_url()
            except ImportError:
                # Final fallback - construct URL from environment or defaults
                host = os.getenv('CLICKHOUSE_HOST', '172.31.6.70')
                port = os.getenv('CLICKHOUSE_PORT', '8123')
                database = os.getenv('CLICKHOUSE_DATABASE', 'prod')
                buffer_size = os.getenv('CLICKHOUSE_BUFFER_SIZE', '50000')
                url = f'http://{host}:{port}/?database={database}&buffer_size={buffer_size}'
        
        product_list = [p.strip() for p in products.split(',')]
        product_filter = f"product IN ({', '.join(repr(p) for p in product_list)})" if len(product_list) > 1 else f"product = '{product_list[0]}'"

        query = f"""
        SELECT *
        FROM ch_recognitions_in
        WHERE {product_filter}
        AND toDate(toTimezone(range_hour, 'Asia/Kolkata')) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
        ORDER BY range_hour
        LIMIT 100000
        FORMAT CSVWithNames
        """

        try:
            response = create_postman_request(url, query)
            if response.status_code != 200:
                print(f"Request failed with status: {response.status_code}")
                print("Response Body:", response.text)
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()

            try:
                df = pd.read_csv(StringIO(response.text))
                
                if df.empty:
                    print("Warning: Query returned no data.")
                    print("This could be due to:")
                    print(f"  - No data for product(s): {products}")
                    print(f"  - No data in date range: {start_date} to {end_date}")
                    print(f"  - Database table 'ch_recognitions_in' might be empty")
                    print(f"  - Product names in database might be different from: {products}")
                    return pd.DataFrame()

                # Rename columns with fallback for different column name variations
                column_mapping = {
                    'product': 'BRAND',
                    'channel': 'CHANNELNAME',
                    'category': 'CATEGORY',
                    'programme': 'PROGRAM NAME',
                    'range_hour': 'AD START TIME',
                    'duration': 'AD DURATION',
                    'creative_language': 'AD LANGUAGE',
                    'creative': 'AD THEME',
                    'brand': 'ADVERTISER NAME'
                }
                
                # Check which columns actually exist and create a mapping
                actual_columns = list(df.columns)
                rename_dict = {}
                
                for db_col, new_col in column_mapping.items():
                    # Try exact match first
                    if db_col in actual_columns:
                        rename_dict[db_col] = new_col
                    # Try case-insensitive match
                    elif db_col.lower() in [col.lower() for col in actual_columns]:
                        matching_col = next(col for col in actual_columns if col.lower() == db_col.lower())
                        rename_dict[matching_col] = new_col
                    else:
                        print(f"Warning: Column '{db_col}' not found in database. Available columns: {actual_columns}")
                
                if rename_dict:
                    df.rename(columns=rename_dict, inplace=True)
                else:
                    print("Warning: No columns could be renamed. Using original column names.")
                
                # Check if required columns exist after renaming
                required_columns = ['BRAND', 'CHANNELNAME', 'CATEGORY', 'AD START TIME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"Error: Missing required columns after renaming: {missing_columns}")
                    print(f"Available columns: {list(df.columns)}")
                    return pd.DataFrame()
                
                # Debug: Check if AD DURATION column was mapped successfully
                if 'AD DURATION' in df.columns:
                    print(f"✅ AD DURATION column found with {len(df['AD DURATION'].dropna())} non-null values")
                    print(f"📊 Sample AD DURATION values: {df['AD DURATION'].dropna().head().tolist()}")
                else:
                    print(f"⚠️ AD DURATION column not found. Available columns: {list(df.columns)}")
                    # Check if duration column exists in original form
                    if 'duration' in df.columns:
                        print(f"📋 Found 'duration' column, renaming to 'AD DURATION'")
                        df.rename(columns={'duration': 'AD DURATION'}, inplace=True)
                    else:
                        print(f"❌ Neither 'AD DURATION' nor 'duration' column found")

                # Parse PROGRAM DATE from AD START TIME
                try:
                    df['PROGRAM DATE'] = pd.to_datetime(df['AD START TIME']).dt.date.astype(str)
                except Exception as e:
                    print(f"Error parsing PROGRAM DATE: {e}")
                    df['PROGRAM DATE'] = df['AD START TIME'].astype(str).str[:10]

                # Duplicate for each region
                if regions:
                    dfs = []
                    for region in regions:
                        temp_df = df.copy()
                        temp_df['region'] = region
                        dfs.append(temp_df)
                    df = pd.concat(dfs, ignore_index=True)

                # Lowercase column names for consistency
                df.columns = df.columns.str.lower()

                # Clean channelname column
                df = clean_column_values(df, 'channelname')

                return df

            except Exception as e:
                print(f"Error processing spot data: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print("All retry attempts failed. Returning empty DataFrame.")
                return pd.DataFrame()
    
    return pd.DataFrame()



