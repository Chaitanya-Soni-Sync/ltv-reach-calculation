"""
Dashboard Data Module
Handles data fetching and processing for the dashboard
"""

import pandas as pd
import requests
from config import get_database_url, get_database_headers


def get_available_products():
    """Get list of available products from database"""
    
    try:
        url = get_database_url()
        headers = get_database_headers()
        
        # Query to get unique products
        query = """
        SELECT DISTINCT brand
        FROM tv_spot_data
        WHERE brand != ''
        ORDER BY brand
        LIMIT 100
        """
        
        response = requests.post(url, headers=headers, data=query, timeout=30)
        
        if response.status_code == 200:
            products = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Add some default products if query returns empty
            if not products:
                products = ["amazon.in", "vim drop", "colgate", "tide", "ariel"]
            
            return products
        else:
            # Fallback products
            return ["amazon.in", "vim drop", "colgate", "tide", "ariel"]
    
    except Exception as e:
        print(f"Error fetching products: {e}")
        # Fallback products
        return ["amazon.in", "vim drop", "colgate", "tide", "ariel"]


def get_available_regions_list():
    """Get list of available regions from database"""
    
    try:
        url = get_database_url()
        headers = get_database_headers()
        
        # Query to get unique regions
        query = """
        SELECT DISTINCT region
        FROM channel_share
        WHERE region != ''
        ORDER BY region
        """
        
        response = requests.post(url, headers=headers, data=query, timeout=30)
        
        if response.status_code == 200:
            regions = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Add default regions if query returns empty
            if not regions:
                regions = ["Delhi", "Chennai", "Mumbai", "Kolkata", "Bangalore", "Hyderabad", "TN/Pondicherry"]
            
            return regions
        else:
            # Fallback regions
            return ["Delhi", "Chennai", "Mumbai", "Kolkata", "Bangalore", "Hyderabad", "TN/Pondicherry"]
    
    except Exception as e:
        print(f"Error fetching regions: {e}")
        # Fallback regions
        return ["Delhi", "Chennai", "Mumbai", "Kolkata", "Bangalore", "Hyderabad", "TN/Pondicherry"]


def validate_date_range(start_date, end_date):
    """Validate date range"""
    
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start > end:
            return False, "Start date must be before end date"
        
        # Check if range is too large (more than 1 year)
        if (end - start).days > 365:
            return False, "Date range cannot exceed 365 days"
        
        return True, "Valid date range"
    
    except Exception as e:
        return False, f"Invalid date format: {str(e)}"


def format_results_for_display(results):
    """Format pipeline results for dashboard display"""
    
    if "error" in results:
        return None
    
    if 'daily_reach' not in results or results['daily_reach'].empty:
        return None
    
    daily_reach = results['daily_reach'].copy()
    
    # Format date
    daily_reach['program date'] = pd.to_datetime(daily_reach['program date']).dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    numeric_columns = ['total_cumulative_spots', 'max_reach', 'total_rating', 'reach_final']
    for col in numeric_columns:
        if col in daily_reach.columns:
            daily_reach[col] = daily_reach[col].round(2)
    
    return daily_reach


def get_summary_stats(results):
    """Calculate summary statistics from results"""
    
    if "error" in results or 'daily_reach' not in results:
        return None
    
    daily_reach = results['daily_reach']
    
    if daily_reach.empty:
        return None
    
    stats = {
        'total_days': len(daily_reach),
        'max_reach': daily_reach['reach_final'].max(),
        'avg_reach': daily_reach['reach_final'].mean(),
        'total_spots': daily_reach['total_cumulative_spots'].max(),
        'total_rating': daily_reach['total_rating'].max(),
        'regions_analyzed': daily_reach['region'].nunique(),
        'products_analyzed': daily_reach['brand'].nunique()
    }
    
    return stats

