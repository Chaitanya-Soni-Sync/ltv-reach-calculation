import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from config import get_database_url, get_database_headers, DATABASE_TABLES

def clean_column_values(df, column_name):
    """Clean column values by removing (tbr), (na), (v) suffixes and trimming whitespace"""
    df[column_name] = df[column_name].astype(str).str.replace(r'\(tbr\)|\(na\)|\(v\)', '', flags=re.IGNORECASE, regex=True).str.strip()
    return df

def normalize_channel_name(channel_name):
    """
    Enhanced channel name normalization for better matching
    
    Args:
        channel_name (str): Raw channel name
        
    Returns:
        str: Normalized channel name
    """
    if pd.isna(channel_name) or channel_name == '':
        return ''
    
    # Convert to string and lowercase
    normalized = str(channel_name).lower().strip()
    
    # Split by '(' and take the first part to remove suffixes like (tbr), (na), (v)
    normalized = normalized.split('(')[0].strip()
    
    # Remove extra whitespace and special characters
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    
    return normalized

def find_best_channel_match(target_channel, available_channels, region):
    """
    Find the best matching channel using intelligent matching
    
    Args:
        target_channel (str): Channel name to match
        available_channels (list): List of available channel names
        region (str): Region for context
        
    Returns:
        str: Best matching channel name or None if no good match
    """
    from difflib import SequenceMatcher
    
    if not target_channel or not available_channels:
        return None
    
    # Step 1: Try exact match first (including HD/SD variations)
    for channel in available_channels:
        if channel.lower().strip() == target_channel.lower().strip():
            return channel
    
    # Step 2: Try normalized match (after removing suffixes)
    normalized_target = normalize_channel_name(target_channel)
    for channel in available_channels:
        if normalize_channel_name(channel) == normalized_target:
            return channel
    
    # Step 3: Handle HD/SD variations intelligently
    # Only normalize HD/SD if the specific HD/SD channel is not available
    target_lower = target_channel.lower().strip()
    
    # Check if target has HD/SD suffix
    if ' hd' in target_lower or ' sd' in target_lower or ' (hd)' in target_lower or ' (sd)' in target_lower:
        # Try to find the base channel name (without HD/SD)
        base_channel = normalize_channel_name(target_channel)
        
        # Look for exact base channel match
        for channel in available_channels:
            if normalize_channel_name(channel) == base_channel:
                return channel
    
    # Step 4: Try fuzzy matching with high threshold as last resort
    best_match = None
    best_ratio = 0.8  # High threshold for fuzzy matching
    
    for channel in available_channels:
        normalized_channel = normalize_channel_name(channel)
        ratio = SequenceMatcher(None, normalized_target, normalized_channel).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = channel
    
    return best_match

def optimize_channel_matching(spot_data_df, channel_info_df):
    """
    Optimize channel matching between spot data and channel share data
    
    Args:
        spot_data_df (pd.DataFrame): Spot data with channel information
        channel_info_df (pd.DataFrame): Channel share data
        
    Returns:
        pd.DataFrame: Optimized merged data with better channel matching
    """
    if spot_data_df.empty or channel_info_df.empty:
        return pd.DataFrame()
    
    # Create copies to avoid modifying original data
    spot_df = spot_data_df.copy()
    channel_df = channel_info_df.copy()
    
    # Normalize channel names in both datasets
    spot_df['channelname_normalized'] = spot_df['channelname'].apply(normalize_channel_name)
    channel_df['channelname_normalized'] = channel_df['channelname'].apply(normalize_channel_name)
    
    # Get unique channels from channel share data for matching
    available_channels = channel_df['channelname'].unique().tolist()
    
    print(f"🔍 Channel matching optimization:")
    print(f"  📊 Spot data channels: {len(spot_df['channelname'].unique())}")
    print(f"  📊 Channel share data channels: {len(available_channels)}")
    
    # Create a mapping for optimized channel matching
    channel_mapping = {}
    unmatched_channels = []
    matched_channels = []
    
    for spot_channel in spot_df['channelname'].unique():
        normalized_spot = normalize_channel_name(spot_channel)
        best_match = find_best_channel_match(spot_channel, available_channels, spot_df['region'].iloc[0])
        
        if best_match:
            channel_mapping[spot_channel] = best_match
            matched_channels.append((spot_channel, best_match))
        else:
            unmatched_channels.append(spot_channel)
    
    # Log matching statistics
    print(f"  ✅ Successfully matched: {len(matched_channels)} channels")
    print(f"  ❌ Unmatched channels: {len(unmatched_channels)}")
    
    if matched_channels:
        print("  📋 Channel mappings:")
        for spot_ch, share_ch in matched_channels[:5]:  # Show first 5 mappings
            spot_normalized = normalize_channel_name(spot_ch)
            share_normalized = normalize_channel_name(share_ch)
            if spot_normalized != share_normalized:
                print(f"    '{spot_ch}' (→ '{spot_normalized}') → '{share_ch}' (→ '{share_normalized}')")
            else:
                print(f"    '{spot_ch}' → '{share_ch}'")
        if len(matched_channels) > 5:
            print(f"    ... and {len(matched_channels) - 5} more")
    
    if unmatched_channels:
        print("  ⚠️ Unmatched channels:")
        for ch in unmatched_channels[:5]:  # Show first 5 unmatched
            normalized = normalize_channel_name(ch)
            print(f"    '{ch}' (normalized: '{normalized}')")
        if len(unmatched_channels) > 5:
            print(f"    ... and {len(unmatched_channels) - 5} more")
    
    # Apply the mapping to spot data
    spot_df['channelname_mapped'] = spot_df['channelname'].map(channel_mapping)
    
    # Create combo keys for both datasets
    spot_df['combo'] = spot_df['region'] + ' - ' + spot_df['channelname_mapped']
    channel_df['combo'] = channel_df['region'] + ' - ' + channel_df['channelname']
    
    # Perform the merge with optimized matching
    merged_df = pd.merge(spot_df, channel_df, on='combo', how='left', suffixes=('_spot', '_info'))
    
    # Debug: Print DataFrame info after merge
    print(f"🔍 DEBUG: merged_df shape after merge: {merged_df.shape}")
    print(f"🔍 DEBUG: merged_df columns after merge: {merged_df.columns.tolist()}")
    print(f"🔍 DEBUG: merged_df head after merge:")
    print(merged_df.head())
    
    # Add region column from spot data for compatibility
    if 'region_spot' in merged_df.columns:
        merged_df['region'] = merged_df['region_spot']
        print(f"🔍 DEBUG: Added 'region' column from 'region_spot'")
    elif 'region_info' in merged_df.columns:
        merged_df['region'] = merged_df['region_info']
        print(f"🔍 DEBUG: Added 'region' column from 'region_info'")
    else:
        print(f"🔴 ERROR: No region column found after merge!")
        return merged_df
    
    # Handle unmatched channels - try region-based fallback
    unmatched_mask = merged_df['channelname_info'].isna()
    if unmatched_mask.any():
        print(f"⚠️ Found {unmatched_mask.sum()} unmatched channels. Attempting region-based fallback...")
        
        # For unmatched channels, try to find any channel in the same region
        for idx in merged_df[unmatched_mask].index:
            region = merged_df.loc[idx, 'region']
            original_channel = merged_df.loc[idx, 'channelname_spot']
            
            # Find any channel in the same region
            region_channels = channel_df[channel_df['region'] == region]
            if not region_channels.empty:
                # Use the first available channel in the region as fallback
                fallback_channel = region_channels.iloc[0]
                merged_df.loc[idx, 'channelname_info'] = fallback_channel['channelname']
                merged_df.loc[idx, 'shr%'] = fallback_channel['shr%']
                merged_df.loc[idx, 'cumeReach%'] = fallback_channel['cumeReach%']
                merged_df.loc[idx, 'ots'] = fallback_channel['ots']
                merged_df.loc[idx, 'rat%'] = fallback_channel['rat%']
                print(f"  🔄 Using fallback channel '{fallback_channel['channelname']}' for '{original_channel}' in {region}")
    
    # Clean up temporary columns
    merged_df = merged_df.drop(['channelname_normalized_spot', 'channelname_normalized_info', 'channelname_mapped'], axis=1, errors='ignore')
    
    # Handle duplicate columns by keeping only the first occurrence
    if 'region' in merged_df.columns:
        # Check if there are duplicate region columns
        region_cols = [col for col in merged_df.columns if col == 'region']
        if len(region_cols) > 1:
            print(f"WARNING: Found {len(region_cols)} duplicate 'region' columns. Removing duplicates...")
            # Keep only the first region column and remove duplicates
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            print(f"🔍 DEBUG: Final merged_df shape: {merged_df.shape}")
            print(f"🔍 DEBUG: Final merged_df columns: {merged_df.columns.tolist()}")
    
    return merged_df

def get_available_regions_from_database(yearmonth: str) -> list:
    """
    Query available regions from channel_share table for a specific month
    
    Args:
        yearmonth (str): Date in format 'YYYY-MM-01' (e.g., '2025-08-01')
        
    Returns:
        list: List of available regions
    """
    try:
        # Construct the query with proper parameter validation
        if not yearmonth or not re.match(r'^\d{4}-\d{2}-\d{2}$', yearmonth):
            print("Invalid yearmonth format. Expected YYYY-MM-DD")
            return []
            
        query = f"""
        SELECT DISTINCT region
        FROM {DATABASE_TABLES['channel_share']} 
        WHERE region != '' 
          AND yearmonth = '{yearmonth}' 
        FORMAT CSVWithNames
        """
        
        # Get database connection details
        db_url = get_database_url()
        headers = get_database_headers()
        
        # Execute query
        response = requests.post(db_url, data=query, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            
            if not df.empty and 'region' in df.columns:
                # Return list of unique regions
                regions = df['region'].unique().tolist()
                # Filter out empty strings and sort
                regions = [region for region in regions if region and str(region).strip()]
                return sorted(regions)
            else:
                print("No region data found in response")
                return []
                
        else:
            print(f"Database query failed with status {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error querying available regions: {e}")
        return []


def get_channel_share_for_date_with_fallback(target_date: str) -> pd.DataFrame:
    """
    Get channel share data for a specific date with intelligent fallback to previous month
    
    Args:
        target_date (str): Target date in format 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: Channel share data for the target date or previous month
    """
    try:
        # Convert target date to datetime
        target_dt = pd.to_datetime(target_date)
        
        # Try to get data for the target month first
        target_yearmonth = target_dt.strftime('%Y-%m-01')
        print(f"🔍 Trying to get channel share data for month: {target_yearmonth}")
        
        channel_data = get_channel_share_from_database(target_yearmonth)
        
        if not channel_data.empty:
            print(f"✅ Successfully retrieved channel share data for {target_yearmonth}")
            return channel_data
        
        # If target month data is not available, try previous month (n-1)
        if target_dt.month == 1:
            # January -> December of previous year
            fallback_yearmonth = f"{target_dt.year - 1}-12-01"
        else:
            # Previous month of same year
            fallback_yearmonth = f"{target_dt.year}-{target_dt.month - 1:02d}-01"
        
        print(f"⚠️ Target month {target_yearmonth} not available, trying fallback: {fallback_yearmonth}")
        
        fallback_data = get_channel_share_from_database(fallback_yearmonth)
        
        if not fallback_data.empty:
            print(f"✅ Successfully retrieved fallback channel share data for {fallback_yearmonth}")
            return fallback_data
        else:
            print(f"❌ Both target month {target_yearmonth} and fallback month {fallback_yearmonth} are not available")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error getting channel share with fallback for date {target_date}: {e}")
        return pd.DataFrame()


def get_channel_share_from_database(yearmonth: str) -> pd.DataFrame:
    """
    Get channel share data for a specific month from the database.
    
    Args:
        yearmonth (str): Date in format 'YYYY-MM-01' (e.g., '2025-08-01')
        
    Returns:
        pd.DataFrame: Channel share data for the specified month.
    """
    try:
        # Construct the query with proper parameter validation
        if not yearmonth or not re.match(r'^\d{4}-\d{2}-\d{2}$', yearmonth):
            print("Invalid yearmonth format. Expected YYYY-MM-DD")
            return pd.DataFrame()
            
        # First, let's check what columns are actually available in the table
        schema_query = f"""
        DESCRIBE {DATABASE_TABLES['channel_share']}
        FORMAT CSVWithNames
        """
        
        db_url = get_database_url()
        headers = get_database_headers()
        
        schema_response = requests.post(db_url, data=schema_query, headers=headers, timeout=30)
        if schema_response.status_code == 200:
            schema_df = pd.read_csv(StringIO(schema_response.text))
            print(f"📋 Database table schema for {DATABASE_TABLES['channel_share']}:")
            for _, row in schema_df.iterrows():
                print(f"  {row['name']}: {row['type']}")
        
        # Use the correct column names based on the data you provided
        query = f"""
        SELECT region, channel, `mf18+_rat_cent`, `mf18+_cume_rch_cent`, `mf18+_shr_cent`, `mf18+_ots`
        FROM {DATABASE_TABLES['channel_share']}
        WHERE yearmonth = '{yearmonth}'
        FORMAT CSVWithNames
        """
        
        print(f"🔍 Querying channel share data for {yearmonth}...")
        
        # Execute query
        response = requests.post(db_url, data=query, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))
            
            if not df.empty and 'region' in df.columns:
                print(f"✅ Retrieved {len(df)} channel share records for {yearmonth}")
                print(f"📊 Sample channels: {df['channel'].unique()[:5].tolist()}")
                
                # Map database columns to expected column names
                column_mapping = {
                    'channel': 'channelname',
                    'mf18+_rat_cent': 'rat%',
                    'mf18+_cume_rch_cent': 'cumeReach%',
                    'mf18+_shr_cent': 'shr%',
                    'mf18+_ots': 'ots'
                }
                
                # Rename columns to match expected format
                df = df.rename(columns=column_mapping)
                
                # Clean column names
                df = clean_column_values(df, 'channelname')
                df = df.assign(channelname=df['channelname'].str.lower(),
                                region=df['region'].str.lower())
                df['combo'] = df['region'] + ' - ' + df['channelname']
                return df
            else:
                print(f"No channel share data found for {yearmonth} in response")
                return pd.DataFrame()
                
        else:
            print(f"Database query failed with status {response.status_code}: {response.text}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error getting channel share from database for {yearmonth}: {e}")
        return pd.DataFrame()


def get_channel_share_with_fallback(target_date: str, max_months_back: int = 12) -> pd.DataFrame:
    """
    Get channel share data with automatic fallback to previous months
    
    Args:
        target_date (str): Target date in YYYY-MM-DD format
        max_months_back (int): Maximum number of months to go back (default: 12)
        
    Returns:
        pd.DataFrame: Channel share data from the most recent available month
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    try:
        # Convert target_date to datetime
        target_dt = pd.to_datetime(target_date)
        
        # Try each month starting from target month, going backwards
        for months_back in range(max_months_back + 1):
            # Calculate the date to try using proper month arithmetic
            if months_back == 0:
                yearmonth = target_dt.strftime('%Y-%m-01')
            else:
                # Go back by months_back months using relativedelta for accurate month calculation
                fallback_date = target_dt - relativedelta(months=months_back)
                yearmonth = fallback_date.strftime('%Y-%m-01')
            
            print(f"🔍 Trying channel share data for {yearmonth}...")
            
            # Try to get data for this month
            df = get_channel_share_from_database(yearmonth)
            
            if not df.empty:
                if months_back == 0:
                    print(f"✅ Found channel share data for target month {yearmonth}")
                else:
                    print(f"✅ Found channel share data for fallback month {yearmonth} (went back {months_back} month{'s' if months_back > 1 else ''})")
                return df
            else:
                print(f"❌ No data found for {yearmonth}")
                continue
        
        # If we get here, no data was found in any of the months
        print(f"❌ No channel share data found for any month within {max_months_back} months of {target_date}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error in channel share fallback mechanism: {e}")
        return pd.DataFrame()


def compute_channel_reach(spot_data_df: pd.DataFrame, channel_info_path: str, target_date: str = None) -> pd.DataFrame:
    # Clean channel names by removing (tbr), (na), (v) suffixes
    spot_data_df = clean_column_values(spot_data_df.copy(), 'channelname')
    
    # Aggregate total spots by region, brand, channelname
    channel_spot_df = spot_data_df.groupby(['region','brand','channelname']).agg({'category':'count'}).reset_index().rename(columns={'category': 'total_spots'}).sort_values(by=['region','brand','total_spots'], ascending=False)
    channel_spot_df = channel_spot_df.assign(channelname=channel_spot_df['channelname'].str.lower(),
                                             region=channel_spot_df['region'].str.lower(),
                                             brand=channel_spot_df['brand'].str.lower())
    channel_spot_df['combo'] = channel_spot_df['region'] + ' - ' + channel_spot_df['channelname']

    # Load channel share info and clean channel names
    if channel_info_path == 'database' and target_date:
        # For database mode, use the fallback mechanism to get channel share data
        try:
            print(f"🔍 Getting channel share data with fallback for {target_date}...")
            channel_info_df = get_channel_share_with_fallback(target_date)
            
            if channel_info_df.empty:
                print(f"❌ No channel share data available for {target_date} or any previous months")
                return pd.DataFrame()
            else:
                print(f"✅ Successfully retrieved channel share data with {len(channel_info_df)} records")
                
        except Exception as e:
            print(f"Error getting channel share data with fallback: {e}")
            return pd.DataFrame()
    else:
        # Fallback to Excel file
        try:
            channel_info_df = pd.read_excel(channel_info_path, header=1)
            channel_info_df.columns = ['region','channelname','rat%','cumeReach%','shr%','ots']
            
            # Clean channel names in channel info data
            channel_info_df = clean_column_values(channel_info_df, 'channelname')
            
            channel_info_df = channel_info_df.assign(channelname=channel_info_df['channelname'].str.lower(),
                                                     region=channel_info_df['region'].str.lower())
            channel_info_df['combo'] = channel_info_df['region'] + ' - ' + channel_info_df['channelname']
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return pd.DataFrame()
    
    # Check if we have channel info data
    if channel_info_df.empty:
        print("No channel share information available")
        return pd.DataFrame()

    # Use optimized channel matching instead of simple merge
    master_channel_df = optimize_channel_matching(channel_spot_df, channel_info_df)

    # Fix 'ots' column: replace 'n.a' with 0 and convert to float
    master_channel_df['ots'] = master_channel_df['ots'].astype(str).replace('n.a', '0', regex=False)
    master_channel_df['ots'] = pd.to_numeric(master_channel_df['ots'], errors='coerce').fillna(0)

    # Calculate rating, ots_f, sync_reach
    master_channel_df['rating'] = (master_channel_df['shr%'] * master_channel_df['total_spots']) / 10
    master_channel_df['ots'] = master_channel_df.apply(
        lambda row: (row['rating'] / row['cumeReach%']) * 10 if row['cumeReach%'] > 0 else 0, axis=1
    )
    master_channel_df['ots_f'] = master_channel_df['ots'].apply(lambda x: x if x > 1 else 0)
    master_channel_df['sync_reach'] = master_channel_df.apply(
        lambda row: row['cumeReach%'] * (1 - np.exp(-0.7 * (row['rating'] / row['cumeReach%'])))
        if row['cumeReach%'] > 0 else 0,
        axis=1
    )

    # Select and rename columns to keep consistent with notebook
    # Only rename columns that exist and don't create duplicates
    rename_dict = {
        'channelname_spot': 'channelname',
        'brand': 'brand',
        'shr%': 'shr%',
        'cumeReach%': 'cumeReach%',
        'ots': 'ots',
        'ots_f': 'ots_f',
        'sync_reach': 'sync_reach',
        'total_spots': 'total_spots'
    }
    
    # Only add region_spot -> region rename if region_spot exists and region doesn't already exist
    if 'region_spot' in master_channel_df.columns and 'region' not in master_channel_df.columns:
        rename_dict['region_spot'] = 'region'
    
    master_channel_df = master_channel_df.rename(columns=rename_dict)

    # Ensure we have a region column
    if 'region' not in master_channel_df.columns:
        print("🔴 ERROR: No region column found after processing!")
        return pd.DataFrame()
    
    # Keep only needed columns
    master_channel_df = master_channel_df[['region', 'channelname', 'brand', 'total_spots', 'shr%', 'cumeReach%', 'ots', 'ots_f', 'sync_reach', 'combo']]
    
    # Ensure no duplicate columns exist
    master_channel_df = master_channel_df.loc[:, ~master_channel_df.columns.duplicated()]
    
    return master_channel_df