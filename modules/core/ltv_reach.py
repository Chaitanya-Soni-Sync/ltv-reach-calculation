#!/usr/bin/env python3
"""
Core LTV Reach Calculation Module
Contains all business logic for LTV reach computations
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from datetime import datetime, timedelta

def compute_ltv_reach_for_campaigns(campaigns_df: pd.DataFrame, regions: List[str]) -> pd.DataFrame:
    """Compute LTV reach for campaigns in specified regions using sophisticated calculation
    
    This function uses the same sophisticated LTV reach calculation as the main dashboard,
    ensuring consistency across all systems.
    
    Args:
        campaigns_df (DataFrame): Campaigns data with brand, region, and reach info
        regions (List[str]): List of regions to process
        
    Returns:
        DataFrame: LTV reach data for campaigns
    """
    try:
        if campaigns_df.empty:
            st.toast("No campaigns data provided for LTV reach computation")
            return pd.DataFrame()
        
        # Import the sophisticated functions from the main modules
        from get_data import get_spot_data
        from channel_reach import compute_channel_reach
        from daily_cum_reach import compute_daily_cumulative_reach
        from config import INPUT_PATHS
        
        # Get date range for consistency
        start_date, end_date = get_real_date_range()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        ltv_results = []
        
        for _, campaign in campaigns_df.iterrows():
            try:
                brand = campaign.get('brand', 'Unknown')
                
                for region in regions:
                    # Get spot data for this brand and region
                    spot_data = get_spot_data(brand, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), [region])
                    
                    if not spot_data.empty:
                        # Compute channel reach using sophisticated calculation
                        start_dt = pd.to_datetime(start_date)
                        channel_reach = compute_channel_reach(spot_data, INPUT_PATHS['channel_share'], start_dt.strftime('%Y-%m-%d'))
                        
                        if not channel_reach.empty:
                            # Compute daily cumulative reach using sophisticated calculation
                            daily_reach = compute_daily_cumulative_reach(spot_data, channel_reach)
                            
                            if not daily_reach.empty:
                                # Get the final reach (same logic as main dashboard)
                                last_date_reach = daily_reach[daily_reach['region'].str.lower() == region.lower()]
                                if not last_date_reach.empty:
                                    # Get the reach value from the maximum date (last day), not the maximum value
                                    max_date = last_date_reach['program date'].max()
                                    final_reach_row = last_date_reach[last_date_reach['program date'] == max_date]
                                    if not final_reach_row.empty:
                                        final_reach = final_reach_row['reach_final'].iloc[0]
                                        total_spots = final_reach_row['total_cumulative_spots'].iloc[0]
                                        total_rating = final_reach_row['total_rating'].iloc[0]
                                    else:
                                        final_reach = 0
                                        total_spots = 0
                                        total_rating = 0
                                else:
                                    final_reach = 0
                                    total_spots = 0
                                    total_rating = 0
                            else:
                                final_reach = 0
                                total_spots = 0
                                total_rating = 0
                        else:
                            final_reach = 0
                            total_spots = 0
                            total_rating = 0
                    else:
                        final_reach = 0
                        total_spots = campaign.get('total_spots', 0)
                        total_rating = 0
                    
                    # Create result row with same structure as main dashboard
                    result_row = {
                        'brand': brand,
                        'ltv_reach': round(final_reach, 2),
                        'region': region,
                        'total_spots': total_spots,
                        'total_rating': round(total_rating, 2),
                        'campaign_score': campaign.get('campaign_score', 0),
                        'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    }
                    
                    ltv_results.append(result_row)
                
            except Exception as e:
                st.error(f"Error processing campaign {brand}: {e}")
                continue
        
        return pd.DataFrame(ltv_results)
        
    except Exception as e:
        st.error(f"Error computing LTV reach: {e}")
        return pd.DataFrame()


def get_top_25_campaigns_last_28_days() -> pd.DataFrame:
    """Get top 25 campaigns from the last 28 days using sophisticated calculation
    
    This function uses the same sophisticated campaign retrieval logic as the main dashboard,
    ensuring consistency across all systems.
    
    Returns:
        DataFrame: Top campaigns data
    """
    try:
        # Import the sophisticated functions from the main modules
        from dashboard import get_top_campaigns_with_real_data
        
        # Use the same sophisticated campaign retrieval as main dashboard
        campaigns = get_top_campaigns_with_real_data()
        
        if not campaigns.empty:
            # Return top 25 campaigns
            return campaigns.head(25)
        else:
            st.warning("No campaigns data available")
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching top campaigns: {e}")
        return pd.DataFrame()


def get_real_date_range() -> Tuple[str, str]:
    """Get real date range for analysis
    
    Returns:
        Tuple[str, str]: Start and end dates in YYYY-MM-DD format
    """
    try:
        # Get current date
        end_date = datetime.now()
        
        # Start date is 28 days ago
        start_date = end_date - timedelta(days=28)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
    except Exception as e:
        st.error(f"Error getting date range: {e}")
        # Fallback dates
        return '2024-01-01', '2024-01-31'


# Note: compute_daily_cumulative_reach and compute_channel_reach functions removed
# These functions now use the sophisticated implementations from the main modules
# to ensure consistency across all systems.
