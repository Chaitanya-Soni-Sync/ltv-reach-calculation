#!/usr/bin/env python3
"""
LTV Reach Computation Pipeline
Computes LTV reach from scratch using the inherited logic from cross_media_dashboard
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from get_data import get_spot_data
from channel_reach import compute_channel_reach
from daily_cum_reach import compute_daily_cumulative_reach
from core.ltv_reach import compute_ltv_reach_for_campaigns, get_top_25_campaigns_last_28_days
from utils import format_dataframe_for_display
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def compute_ltv_reach_pipeline(products: str, start_date: str, end_date: str, regions: list = None):
    """
    Complete LTV reach computation pipeline from scratch
    
    Args:
        products (str): Product name(s) to analyze
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        regions (list): List of regions to analyze (optional)
    
    Returns:
        dict: Results containing spot data, channel reach, daily reach, and LTV reach
    """
    print(f"🚀 Starting LTV Reach Pipeline")
    print(f"  Products: {products}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Regions: {regions if regions else 'All available'}")
    
    # Create output folder if not exists
    if not os.path.exists('output'):
        os.makedirs('output')
    
    results = {}
    
    try:
        # Step 1: Get Spot Data
        print("\n--- Step 1: Fetching Spot Data ---")
        spot_data_df = get_spot_data(products, start_date, end_date, regions)
        
        if spot_data_df.empty:
            print("❌ No spot data fetched. Pipeline cannot continue.")
            return {"error": "No spot data available"}
        
        print(f"✅ Successfully fetched {len(spot_data_df)} spot records")
        results['spot_data'] = spot_data_df
        
        # Step 2: Channel Reach
        print("\n--- Step 2: Computing Channel Reach ---")
        from config import INPUT_PATHS
        start_dt = pd.to_datetime(start_date)
        master_channel_df = compute_channel_reach(spot_data_df, INPUT_PATHS['channel_share'], start_dt.strftime('%Y-%m-%d'))
        
        if master_channel_df.empty:
            print("❌ Channel reach computation failed")
            return {"error": "Channel reach computation failed"}
        
        print(f"✅ Channel reach computed: {len(master_channel_df)} records")
        results['channel_reach'] = master_channel_df
        
        # Step 3: Daily Cumulative Reach
        print("\n--- Step 3: Computing Daily Cumulative Reach ---")
        daily_reach_computed = compute_daily_cumulative_reach(spot_data_df, master_channel_df)
        
        if daily_reach_computed.empty:
            print("❌ Daily cumulative reach computation failed")
            return {"error": "Daily cumulative reach computation failed"}
        
        print(f"✅ Daily cumulative reach computed: {len(daily_reach_computed)} records")
        results['daily_reach'] = daily_reach_computed
        
        # Step 4: LTV Reach Analysis (Final reach from daily cumulative reach)
        print("\n--- Step 4: Computing LTV Reach ---")
        
        # LTV reach is the final reach from daily cumulative reach (same as reference project)
        ltv_data = []
        for region in daily_reach_computed['region'].unique():
            for brand in daily_reach_computed['brand'].unique():
                brand_region_data = daily_reach_computed[
                    (daily_reach_computed['region'] == region) & 
                    (daily_reach_computed['brand'] == brand)
                ]
                
                if not brand_region_data.empty:
                    # Get the final reach value (last day's reach) - this IS the LTV reach
                    final_reach = brand_region_data['reach_final'].iloc[-1]
                    total_spots = brand_region_data['total_cumulative_spots'].iloc[-1]
                    total_rating = brand_region_data['total_rating'].iloc[-1]
                    max_reach = brand_region_data['max_reach'].iloc[-1]
                    
                    ltv_data.append({
                        'brand': brand,
                        'ltv_reach': round(final_reach, 2),
                        'region': region,
                        'total_spots': total_spots,
                        'total_rating': round(total_rating, 2),
                        'max_reach': round(max_reach, 2),
                        'campaign_type': 'low_frequency' if max_reach < 40 else 'high_frequency',
                        'date_range': f"{start_date} to {end_date}"
                    })
        
        results['ltv_reach'] = pd.DataFrame(ltv_data)
        print(f"✅ LTV reach computed: {len(ltv_data)} records")
        
        # Save results to CSV files
        print("\n--- Saving Results ---")
        spot_data_df.to_csv('output/spot_data.csv', index=False)
        master_channel_df.to_csv('output/channel_reach.csv', index=False)
        daily_reach_computed.to_csv('output/daily_cumulative_reach.csv', index=False)
        
        if not results['ltv_reach'].empty:
            results['ltv_reach'].to_csv('output/ltv_reach_results.csv', index=False)
        
        print("✅ All results saved to 'output' folder")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in LTV reach pipeline: {e}")
        return {"error": str(e)}


def display_ltv_results(results: dict):
    """
    Display only daily LTV reach results in tabular format
    
    Args:
        results (dict): Results from compute_ltv_reach_pipeline
    """
    if "error" in results:
        print(f"❌ Pipeline Error: {results['error']}")
        return
    
    print("\n" + "="*80)
    print("📊 DAILY LTV REACH RESULTS")
    print("="*80)
    
    # Display only Daily Reach (LTV Reach)
    if 'daily_reach' in results:
        daily_reach = results['daily_reach']
        
        # Prepare display DataFrame with renamed columns (same as reference project)
        display_df = daily_reach.rename(columns={
            'region': 'Region',
            'program date': 'Date',
            'brand': 'Product',
            'total_cumulative_spots': 'Cumm_Spots',
            'max_reach': 'Max Reach',
            'total_rating': 'Rating',
            'reach_computed': 'computed_Reach',
            'reach_final': 'LTV_Reach'
        })
        display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Display columns for LTV reach
        display_columns = ['Region', 'Date', 'Product', 'Cumm_Spots', 'Max Reach', 'Rating', 'LTV_Reach']
        
        # Format dataframe with proper number formatting and row numbering
        display_df_formatted = format_dataframe_for_display(
            display_df[display_columns], 
            reset_index=True, 
            round_decimals=2
        )
        
        print(f"\n📈 DAILY LTV REACH TABLE")
        print("-" * 100)
        print(display_df_formatted.to_string())
        
        # Summary statistics
        print(f"\n📊 LTV REACH SUMMARY")
        print(f"  Total Days: {len(daily_reach)}")
        print(f"  Max LTV Reach Achieved: {daily_reach['reach_final'].max():.2f}%")
        print(f"  Average Daily LTV Reach: {daily_reach['reach_final'].mean():.2f}%")
        
        # Final LTV reach by region and brand
        final_ltv_summary = daily_reach.groupby(['region', 'brand']).agg({
            'reach_final': 'last',
            'total_cumulative_spots': 'last',
            'total_rating': 'last',
            'max_reach': 'last'
        }).reset_index()
        
        print(f"\n🎯 FINAL LTV REACH BY REGION & BRAND")
        print("-" * 80)
        formatted_summary = format_dataframe_for_display(
            final_ltv_summary.rename(columns={
                'region': 'Region',
                'brand': 'Brand', 
                'reach_final': 'Final LTV Reach (%)',
                'total_cumulative_spots': 'Total Spots',
                'total_rating': 'Total Rating',
                'max_reach': 'Max Reach (%)'
            }), 
            reset_index=True, 
            round_decimals=2
        )
        print(formatted_summary.to_string())
    
    print("\n" + "="*80)
    print("✅ DAILY LTV REACH COMPUTATION COMPLETED")
    print("="*80)


def main():
    """Main function to run LTV reach computation"""
    
    # Example parameters - modify as needed
    products = "vim drop"  # Change to your product name
    start_date = "2025-01-01"  # Change to your start date
    end_date = "2025-01-31"    # Change to your end date
    regions = ['Delhi', 'TN/Pondicherry']  # Change to your regions or None for all
    
    print("🎯 LTV REACH COMPUTATION FROM SCRATCH")
    print("="*50)
    
    # Run the complete pipeline
    results = compute_ltv_reach_pipeline(products, start_date, end_date, regions)
    
    # Display results in tabular format
    display_ltv_results(results)
    
    return results


if __name__ == "__main__":
    main()