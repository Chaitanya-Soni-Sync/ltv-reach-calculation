import pandas as pd
import numpy as np
import warnings
import re

# Suppress pandas FutureWarning for groupby.apply
warnings.filterwarnings('ignore', category=FutureWarning, message='.*groupby.*apply.*')

def clean_column_values(df, column_name):
    """Clean column values by removing (tbr), (na), (v) suffixes and trimming whitespace"""
    df[column_name] = df[column_name].astype(str).str.replace(r'\(tbr\)|\(na\)|\(v\)', '', flags=re.IGNORECASE, regex=True).str.strip()
    return df


def calculate_frequency_adjusted_ltv_reach(spot_data_df: pd.DataFrame, ltv_reach_value: float, campaign_frequency_type: str = "high") -> float:
    """
    Calculate frequency-adjusted LTV reach based on campaign characteristics
    
    Args:
        spot_data_df (pd.DataFrame): Spot data containing duration information
        ltv_reach_value (float): Original LTV reach value
        campaign_frequency_type (str): Campaign frequency type ("high" or "low")
        
    Returns:
        float: Frequency-adjusted LTV reach value
        
    Logic:
    - High frequency campaigns: Apply 83% multiplier to ALL LTV reach values (regardless of duration)
    - Low frequency campaigns: Apply 125% multiplier ONLY if 15-second creative distribution > 75%
    """
    try:
        if spot_data_df.empty or ltv_reach_value <= 0:
            return ltv_reach_value
        
        if campaign_frequency_type.lower() == "high":
            # High frequency campaign - apply 83% multiplier regardless of duration
            adjusted_reach = ltv_reach_value * 0.83
            print(f"✅ High Frequency Campaign:")
            print(f"  📉 Applying 83% multiplier to ALL LTV reach (regardless of duration)")
            print(f"  📊 Original LTV Reach: {ltv_reach_value:.2f}% → Adjusted: {adjusted_reach:.2f}%")
            
        else:
            # Low frequency campaign - check duration distribution
            print(f"🔍 Available columns in spot data: {list(spot_data_df.columns)}")
            if 'AD DURATION' not in spot_data_df.columns:
                print("⚠️ Duration column not found for low frequency campaign, using original LTV reach value")
                return ltv_reach_value
            
            # Clean and process duration data
            duration_data = spot_data_df['AD DURATION'].dropna()
            if duration_data.empty:
                print("⚠️ No duration data available for low frequency campaign, using original LTV reach value")
                return ltv_reach_value
            
            # Convert duration to numeric (assuming it's in seconds)
            try:
                duration_numeric = pd.to_numeric(duration_data, errors='coerce').dropna()
            except Exception as e:
                print(f"⚠️ Error converting duration to numeric: {e}")
                return ltv_reach_value
            
            if duration_numeric.empty:
                print("⚠️ No valid duration data after conversion for low frequency campaign, using original LTV reach value")
                return ltv_reach_value
            
            # Calculate percentage of 15-second creatives
            fifteen_sec_spots = duration_numeric[duration_numeric == 15].count()
            total_spots = duration_numeric.count()
            
            if total_spots == 0:
                print("⚠️ No spots found for duration analysis in low frequency campaign, using original LTV reach value")
                return ltv_reach_value
            
            fifteen_sec_percentage = (fifteen_sec_spots / total_spots) * 100
            
            print(f"🔍 Low Frequency Campaign - Duration Analysis:")
            print(f"  📊 Total spots: {total_spots}")
            print(f"  📊 15-second spots: {fifteen_sec_spots}")
            print(f"  📊 15-second percentage: {fifteen_sec_percentage:.2f}%")
            
            if fifteen_sec_percentage > 75:
                # Low frequency with >75% 15s creatives - apply 125% multiplier
                adjusted_reach = ltv_reach_value * 1.25
                print(f"✅ Low frequency campaign with >75% 15s creatives detected")
                print(f"  📈 Applying 125% multiplier: {ltv_reach_value:.2f}% → {adjusted_reach:.2f}%")
            else:
                # Low frequency with ≤75% 15s creatives - no adjustment
                adjusted_reach = ltv_reach_value
                print(f"✅ Low frequency campaign with ≤75% 15s creatives detected")
                print(f"  📊 No adjustment applied: {ltv_reach_value:.2f}%")
        
        # Cap at 100%
        final_reach = min(adjusted_reach, 100.0)
        
        if final_reach != adjusted_reach:
            print(f"  🚫 Capped at 100%: {adjusted_reach:.2f}% → {final_reach:.2f}%")
        
        return final_reach
        
    except Exception as e:
        print(f"❌ Error in frequency adjustment calculation: {e}")
        return ltv_reach_value

"""
TV Spot Daily Cumulative Reach Calculator

This module implements the daily cumulative reach calculation for TV advertising campaigns.
Both high-frequency and low-frequency campaigns now use the same Enhanced Progressive Growth Logic.

Recent Updates (Unified Enhanced Progressive Growth Logic):
- Enhanced progressive growth logic for ALL campaigns (both high and low GRP)
- Progressive reach growth at intervals of max_reach*10 with 4% increments
- Smart interpolation between key points based on rating progression
- Ensures cumulative reach growth (never decreases)
- Unified approach for consistent reach modeling across all campaign types

Enhanced Progressive Growth Logic (All Campaigns):
1. Before rating < 58.4: reach = computed reach
2. At rating = 58.4: next day reach = max_reach
3. At rating = 116.8: reach = max_reach * 1.04 (4% growth)
4. At rating = 175.2: reach = (max_reach * 1.04) * 1.08 (8% growth)
5. At rating = 233.6: reach = ((max_reach * 1.04) * 1.08) * 1.12 (12% growth)
6. And so on... with increasing growth rates at each 58.4 rating interval
6. Intermediate days interpolated proportionally based on rating increase

GRP Factor Mapping (legacy - now used for reach factor calculation only):
  * 0-1000: 1.02x
  * 1000-1500: 1.04x  
  * 1500-2000: 1.06x
  * 2000-2500: 1.05x
  * 2500-3000: 1.04x
  * 3000-3500: 1.03x
  * 3500-4000: 1.02x
  * 4000+: 1.01x
- Fallback to 1.0x factor if no matching GRP range is found
"""


def compute_daily_cumulative_reach(spot_data_df: pd.DataFrame, master_channel_df: pd.DataFrame) -> pd.DataFrame:
    # Prepare and clean data
    daily_spot_data = spot_data_df.copy()
    
    # Clean channel names by removing (tbr), (na), (v) suffixes
    daily_spot_data = clean_column_values(daily_spot_data, 'channelname')
    
    daily_spot_data['program date'] = pd.to_datetime(daily_spot_data['program date'])
    for col in ['region', 'channelname', 'brand']:
        daily_spot_data[col] = daily_spot_data[col].str.lower()

    # Aggregate spots per region, brand, channel, date
    base_channel_spot_daily = (
        daily_spot_data.groupby(['region', 'brand', 'channelname', 'program date'])
        .agg(spots=('category', 'count'))
        .reset_index()
    )

    # Create full date range and all region-brand-channel combos
    all_dates = pd.date_range(base_channel_spot_daily['program date'].min(), base_channel_spot_daily['program date'].max())
    combos = base_channel_spot_daily[['region', 'brand', 'channelname']].drop_duplicates()
    full_index = combos.merge(pd.DataFrame({'program date': all_dates}), how='cross')

    # Merge spots, fill missing with 0
    full_daily_data = full_index.merge(base_channel_spot_daily, on=['region', 'brand', 'channelname', 'program date'], how='left')
    full_daily_data['spots'] = full_daily_data['spots'].fillna(0)
    full_daily_data = full_daily_data.sort_values(['region', 'brand', 'channelname', 'program date'])

    # Calculate cumulative spots per channel
    full_daily_data['cumulative_spots_channel'] = full_daily_data.groupby(['region', 'brand', 'channelname'])['spots'].cumsum()

    # Check if master_channel_df has the required columns
    if master_channel_df.empty:
        print("Warning: master_channel_df is empty, cannot compute daily cumulative reach")
        return pd.DataFrame()
    
    required_columns = ['region', 'channelname', 'shr%', 'cumeReach%']
    missing_columns = [col for col in required_columns if col not in master_channel_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in master_channel_df: {missing_columns}")
        print(f"Available columns: {master_channel_df.columns.tolist()}")
        return pd.DataFrame()
    
    # Debug: Print DataFrames before merge
    print(f"🔍 DEBUG: full_daily_data shape: {full_daily_data.shape}")
    print(f"🔍 DEBUG: full_daily_data columns: {full_daily_data.columns.tolist()}")
    print(f"🔍 DEBUG: full_daily_data head:")
    print(full_daily_data.head())
    
    print(f"🔍 DEBUG: master_channel_df shape: {master_channel_df.shape}")
    print(f"🔍 DEBUG: master_channel_df columns: {master_channel_df.columns.tolist()}")
    print(f"🔍 DEBUG: master_channel_df head:")
    print(master_channel_df.head())
    
    # Handle duplicate columns in master_channel_df before selection
    master_channel_df = master_channel_df.loc[:, ~master_channel_df.columns.duplicated()]
    
    # Merge channel info (shr%, cumeReach%)
    channel_info = master_channel_df[['region', 'channelname', 'shr%', 'cumeReach%']].drop_duplicates()
    
    print(f"🔍 DEBUG: channel_info shape: {channel_info.shape}")
    print(f"🔍 DEBUG: channel_info columns: {channel_info.columns.tolist()}")
    print(f"🔍 DEBUG: channel_info head:")
    print(channel_info.head())
    
    merged = full_daily_data.merge(channel_info, on=['region', 'channelname'], how='left')
    merged['shr%'] = merged['shr%'].fillna(0)
    merged['cumeReach%'] = merged['cumeReach%'].fillna(0)

    # Calculate daily channel rating
    merged['rating_channel'] = (merged['shr%'] * merged['cumulative_spots_channel']) / 10

    # Aggregate to region-brand-date level
    daily_reach = merged.groupby(['region', 'brand', 'program date']).agg(
        total_rating_sum=('rating_channel', 'sum'),
        total_cumulative_spots=('cumulative_spots_channel', 'sum')
    ).reset_index()

    daily_reach = daily_reach.sort_values(['region', 'brand', 'program date'])
    daily_reach['total_rating'] = daily_reach.groupby(['region', 'brand'])['total_rating_sum'].cummax()
    daily_reach['total_cumulative_spots'] = daily_reach.groupby(['region', 'brand'])['total_cumulative_spots'].cummax()

    # Get max reach per region-brand
    max_reach_df = master_channel_df.groupby(['region', 'brand']).agg(max_reach=('shr%', 'sum')).reset_index()
    daily_reach = daily_reach.merge(max_reach_df, on=['region', 'brand'], how='left')
    daily_reach['max_reach'] = daily_reach['max_reach'].fillna(0)

    # Compute original reach
    daily_reach['reach_computed'] = daily_reach['max_reach'] * (1 - np.exp(-0.007 * daily_reach['total_rating']))

    # Load GRP based reach factor
    from config import INPUT_PATHS
    grp_reach_factor_df = pd.read_csv(INPUT_PATHS['grp_factors'])
    
    # Convert to numeric to ensure proper comparison
    grp_reach_factor_df['min_grp'] = pd.to_numeric(grp_reach_factor_df['min_grp'])
    grp_reach_factor_df['max_grp'] = pd.to_numeric(grp_reach_factor_df['max_grp'])
    grp_reach_factor_df['factor_over70_percent_reach'] = pd.to_numeric(grp_reach_factor_df['factor_over70_percent_reach'])
    grp_reach_factor_df['factor when rating_equals_max_reach%'] = pd.to_numeric(grp_reach_factor_df['factor when rating_equals_max_reach%'])

    def get_reach_factor(grp):
        grp = float(grp)
        # Use inclusive range check (>= min and <= max)
        row = grp_reach_factor_df[(grp_reach_factor_df['min_grp'] <= grp) & (grp_reach_factor_df['max_grp'] >= grp)]
        if not row.empty:
            factor = row['factor_over70_percent_reach'].values[0]
            return factor
        else:
            return 1.0

    def get_grp_factor(grp):
        """Get GRP factor for low GRP campaigns based on rating value"""
        grp = float(grp)
        # Use inclusive range check (>= min and <= max)
        row = grp_reach_factor_df[(grp_reach_factor_df['min_grp'] <= grp) & (grp_reach_factor_df['max_grp'] >= grp)]
        if not row.empty:
            factor = row['factor when rating_equals_max_reach%'].values[0]
            return factor
        else:
            return 1.0  # Default factor if no matching GRP range is found

    def process_campaign(group):
        max_reach = group['max_reach'].iloc[0]
        group = group.copy()
        
        # Store original computed reach
        group['reach_original'] = group['reach_computed']
        
        # Add reach factor column for all campaigns
        group['reach_factor'] = group['total_rating'].apply(get_reach_factor)
        

        
        if max_reach >= 40:  # High frequency campaign - Now using Enhanced Progressive Growth Logic
            ratings = group['total_rating'].values
            reach = group['reach_computed'].values
            n = len(group)
            
            reach_final = np.full(n, np.nan)
            
            # 1. Before rating < 58.4: updated reach = computed reach
            before_max_mask = ratings < (max_reach*10)
            reach_final[before_max_mask] = reach[before_max_mask]
            
            # 2. Find first day rating crosses or equals 58.4
            idx_cross_candidates = np.where(ratings >= max_reach*10)[0]
            if len(idx_cross_candidates) == 0:
                group['reach_final'] = reach_final
                group['campaign_type'] = 'high_frequency'
                return group
            idx_cross = idx_cross_candidates[0]
            
            # 3. On next day after crossing 58.4, updated reach = max_reach
            reach_final[idx_cross] = reach[idx_cross]  # crossing day: original reach
            if idx_cross + 1 < n:
                reach_final[idx_cross + 1] = max_reach
            
            # 4. Enhanced Progressive Growth Logic for High GRP Campaigns
            # Apply progressive growth at intervals of 58.4 with increasing growth rates
            current_reach = max_reach  # Start from max_reach after crossing point
            current_rating_threshold = max_reach*10  # Initial threshold
            growth_cycle = 0  # Track which growth cycle we're in
            
            # Track key points for interpolation
            key_points = [(idx_cross + 1, max_reach)]  # First key point after crossing
            
            # Find all key points where rating crosses 58.4, 116.8, 175.2, etc.
            i = idx_cross + 2
            while i < n:
                # Check if we've reached the next threshold
                next_threshold = current_rating_threshold + max_reach*10
                
                if ratings[i] >= next_threshold:
                    # We've crossed the next threshold, apply increasing growth rate
                    growth_cycle += 1
                    growth_rate = 1.0 + (0.04 * growth_cycle)  # 4%, 8%, 12%, 16%, etc.
                    current_reach = current_reach * growth_rate
                    current_rating_threshold = next_threshold
                    key_points.append((i, current_reach))
                    
                    # Fill reach for this day
                    reach_final[i] = current_reach
                    
                    # Interpolate between previous key point and this one
                    if len(key_points) >= 2:
                        prev_idx, prev_reach = key_points[-2]
                        curr_idx, curr_reach = key_points[-1]
                        
                        # Fill intermediate days proportionally based on rating increase
                        for j in range(prev_idx + 1, curr_idx):
                            if j < n:
                                # Calculate proportional reach based on rating position between thresholds
                                prev_rating = ratings[prev_idx]
                                curr_rating = ratings[curr_idx]
                                current_rating = ratings[j]
                                
                                if curr_rating > prev_rating:
                                    # Linear interpolation based on rating position
                                    rating_ratio = (current_rating - prev_rating) / (curr_rating - prev_rating)
                                    interpolated_reach = prev_reach + (rating_ratio * (curr_reach - prev_reach))
                                    reach_final[j] = interpolated_reach
                                else:
                                    # If rating didn't increase, maintain previous reach
                                    reach_final[j] = prev_reach
                    
                    i += 1
                else:
                    # We haven't crossed the next threshold yet
                    # Check if we're between key points and need to interpolate
                    if len(key_points) >= 1:
                        last_key_idx, last_key_reach = key_points[-1]
                        
                        # If we're between the last key point and the next threshold
                        if ratings[i] > ratings[last_key_idx]:
                            # Interpolate based on rating increase from last key point
                            last_rating = ratings[last_key_idx]
                            next_threshold = current_rating_threshold + max_reach*10
                            
                            if next_threshold > last_rating:
                                # Calculate proportional reach based on rating position
                                rating_ratio = (ratings[i] - last_rating) / (next_threshold - last_rating)
                                # Estimate next reach with increasing growth rate
                                next_growth_cycle = growth_cycle + 1
                                next_growth_rate = 1.0 + (0.04 * next_growth_cycle)  # 4%, 8%, 12%, 16%, etc.
                                estimated_next_reach = last_key_reach * next_growth_rate
                                interpolated_reach = last_key_reach + (rating_ratio * (estimated_next_reach - last_key_reach))
                                reach_final[i] = interpolated_reach
                            else:
                                # Maintain last key reach if rating hasn't increased significantly
                                reach_final[i] = last_key_reach
                        else:
                            # Rating didn't increase, maintain last key reach
                            reach_final[i] = last_key_reach
                    else:
                        # No key points yet, maintain max_reach
                        reach_final[i] = max_reach
                    
                    i += 1
            
            # Fill any remaining NaNs with appropriate values
            for k in range(n):
                if np.isnan(reach_final[k]):
                    if k == 0:
                        reach_final[k] = reach[k]
                    else:
                        # Use previous day's reach or computed reach, whichever is higher (cumulative nature)
                        prev_reach = reach_final[k-1] if not np.isnan(reach_final[k-1]) else reach[k-1]
                        reach_final[k] = max(prev_reach, reach[k])

            # Post-process to remove flat curves and ensure cumulative growth
            # Find the first index where the reach stagnates
            stagnation_start_idx = -1
            for k in range(1, n):
                # Using a small threshold for floating point comparison
                if abs(reach_final[k] - reach_final[k-1]) < 1e-6:
                    stagnation_start_idx = k - 1
                    break
            
            if stagnation_start_idx != -1:
                num_steps = n - stagnation_start_idx
                if num_steps > 1:
                    start_val = reach_final[stagnation_start_idx]
                    
                    # For high GRP campaigns, use a more conservative approach
                    # Calculate target based on progressive growth pattern
                    if max_reach >= 40:  # High GRP campaign
                        # Estimate final reach based on progressive growth
                        rating_progress = ratings[-1] / (max_reach * 10) if max_reach > 0 else 0
                        if rating_progress > 1:
                            # Calculate how many 4% increments we should have
                            num_increments = int(rating_progress - 1)
                            target_reach = max_reach * (1.04 ** num_increments)
                            target_reach = min(target_reach, 100)  # Cap at 100%
                        else:
                            target_reach = max_reach
                    else:
                        # Use reach_adjusted as the target end value for low frequency
                        reach_adjusted = group['reach_computed'].values * group['reach_factor'].values
                        target_reach = reach_adjusted[-1]
                    
                    # Only apply if it creates a growing curve
                    if target_reach > start_val:
                        new_segment = np.linspace(start_val, target_reach, num=num_steps)
                        reach_final[stagnation_start_idx:] = new_segment

            # cap at 100%
            reach_final = np.minimum(reach_final, 100)
            
            group['reach_final'] = reach_final
            group['campaign_type'] = 'high_frequency'
            
        else:  # Low frequency campaign
            ratings = group['total_rating'].values
            reach = group['reach_computed'].values
            n = len(group)
            
            reach_final = np.full(n, np.nan)
            
            final_rating = ratings[-1]
            half_final_rating = 0.5 * final_rating
            
            # 1. Before rating < 58.4: updated reach = computed reach
            before_max_mask = ratings < (max_reach*10)
            reach_final[before_max_mask] = reach[before_max_mask]
            
            # 2. Find first day rating crosses or equals 58.4
            idx_cross_candidates = np.where(ratings >= max_reach*10)[0]
            if len(idx_cross_candidates) == 0:
                group['reach_final'] = reach_final
                group['campaign_type'] = 'low_frequency'
                return group
            idx_cross = idx_cross_candidates[0]
            
            # 3. On next day after crossing 58.4, updated reach = max_reach
            reach_final[idx_cross] = reach[idx_cross]  # crossing day: original reach
            if idx_cross + 1 < n:
                reach_final[idx_cross + 1] = max_reach
            
            # 4. Enhanced Progressive Growth Logic for Low GRP Campaigns
            # Apply progressive growth at intervals of 58.4 with increasing growth rates
            current_reach = max_reach  # Start from max_reach after crossing point
            current_rating_threshold = max_reach*10  # Initial threshold
            growth_cycle = 0  # Track which growth cycle we're in
            
            # Track key points for interpolation
            key_points = [(idx_cross + 1, max_reach)]  # First key point after crossing
            
            # Find all key points where rating crosses 58.4, 116.8, 175.2, etc.
            i = idx_cross + 2
            while i < n:
                # Check if we've reached the next threshold
                next_threshold = current_rating_threshold + max_reach*10
                
                if ratings[i] >= next_threshold:
                    # We've crossed the next threshold, apply increasing growth rate
                    growth_cycle += 1
                    growth_rate = 1.0 + (0.04 * growth_cycle)  # 4%, 8%, 12%, 16%, etc.
                    current_reach = current_reach * growth_rate
                    current_rating_threshold = next_threshold
                    key_points.append((i, current_reach))
                    
                    # Fill reach for this day
                    reach_final[i] = current_reach
                    
                    # Interpolate between previous key point and this one
                    if len(key_points) >= 2:
                        prev_idx, prev_reach = key_points[-2]
                        curr_idx, curr_reach = key_points[-1]
                        
                        # Fill intermediate days proportionally based on rating increase
                        for j in range(prev_idx + 1, curr_idx):
                            if j < n:
                                # Calculate proportional reach based on rating position between thresholds
                                prev_rating = ratings[prev_idx]
                                curr_rating = ratings[curr_idx]
                                current_rating = ratings[j]
                                
                                if curr_rating > prev_rating:
                                    # Linear interpolation based on rating position
                                    rating_ratio = (current_rating - prev_rating) / (curr_rating - prev_rating)
                                    interpolated_reach = prev_reach + (rating_ratio * (curr_reach - prev_reach))
                                    reach_final[j] = interpolated_reach
                                else:
                                    # If rating didn't increase, maintain previous reach
                                    reach_final[j] = prev_reach
                    
                    i += 1
                else:
                    # We haven't crossed the next threshold yet
                    # Check if we're between key points and need to interpolate
                    if len(key_points) >= 1:
                        last_key_idx, last_key_reach = key_points[-1]
                        
                        # If we're between the last key point and the next threshold
                        if ratings[i] > ratings[last_key_idx]:
                            # Interpolate based on rating increase from last key point
                            last_rating = ratings[last_key_idx]
                            next_threshold = current_rating_threshold + max_reach*10
                            
                            if next_threshold > last_rating:
                                # Calculate proportional reach based on rating position
                                rating_ratio = (ratings[i] - last_rating) / (next_threshold - last_rating)
                                # Estimate next reach with increasing growth rate
                                next_growth_cycle = growth_cycle + 1
                                next_growth_rate = 1.0 + (0.04 * next_growth_cycle)  # 4%, 8%, 12%, 16%, etc.
                                estimated_next_reach = last_key_reach * next_growth_rate
                                interpolated_reach = last_key_reach + (rating_ratio * (estimated_next_reach - last_key_reach))
                                reach_final[i] = interpolated_reach
                            else:
                                # Maintain last key reach if rating hasn't increased significantly
                                reach_final[i] = last_key_reach
                        else:
                            # Rating didn't increase, maintain last key reach
                            reach_final[i] = last_key_reach
                    else:
                        # No key points yet, maintain max_reach
                        reach_final[i] = max_reach
                    
                    i += 1
            
            # Fill any remaining NaNs with appropriate values
            for k in range(n):
                if np.isnan(reach_final[k]):
                    if k == 0:
                        reach_final[k] = reach[k]
                    else:
                        # Use previous day's reach or computed reach, whichever is higher (cumulative nature)
                        prev_reach = reach_final[k-1] if not np.isnan(reach_final[k-1]) else reach[k-1]
                        reach_final[k] = max(prev_reach, reach[k])

            # Post-process to remove flat curves and ensure cumulative growth
            # Find the first index where the reach stagnates
            stagnation_start_idx = -1
            for k in range(1, n):
                # Using a small threshold for floating point comparison
                if abs(reach_final[k] - reach_final[k-1]) < 1e-6:
                    stagnation_start_idx = k - 1
                    break
            
            if stagnation_start_idx != -1:
                num_steps = n - stagnation_start_idx
                if num_steps > 1:
                    start_val = reach_final[stagnation_start_idx]
                    
                    # For low GRP campaigns, use a more conservative approach
                    # Calculate target based on progressive growth pattern
                    if max_reach < 40:  # Low GRP campaign
                        # Estimate final reach based on progressive growth
                        rating_progress = ratings[-1] / (max_reach * 10) if max_reach > 0 else 0
                        if rating_progress > 1:
                            # Calculate how many 4% increments we should have
                            num_increments = int(rating_progress - 1)
                            target_reach = max_reach * (1.04 ** num_increments)
                            target_reach = min(target_reach, 100)  # Cap at 100%
                        else:
                            target_reach = max_reach
                    else:
                        # Use reach_adjusted as the target end value for high frequency
                        reach_adjusted = group['reach_computed'].values * group['reach_factor'].values
                        target_reach = reach_adjusted[-1]
                    
                    # Only apply if it creates a growing curve
                    if target_reach > start_val:
                        new_segment = np.linspace(start_val, target_reach, num=num_steps)
                        reach_final[stagnation_start_idx:] = new_segment

            # cap at 100%
            reach_final = np.minimum(reach_final, 100)
            
            group['reach_final'] = reach_final
            group['campaign_type'] = 'low_frequency'
        
        return group

    # Process each campaign group once with the appropriate logic
    # Use a different approach to avoid the pandas warning
    daily_reach = daily_reach.groupby(['region', 'brand']).apply(process_campaign).reset_index(drop=True)
    
    # Fill any remaining NaNs
    daily_reach['reach_final'] = daily_reach['reach_final'].fillna(daily_reach['reach_computed'])
    
    # DO NOT add updated_Reach column here - it will be added in dashboard.py
    # daily_reach['updated_Reach'] = daily_reach['reach_final']
    
    return daily_reach
