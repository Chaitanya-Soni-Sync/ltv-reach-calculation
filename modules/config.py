"""
Configuration file for Cross-Media Dashboard
Centralizes all configuration values and constants
"""

import os
from typing import List

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('CLICKHOUSE_HOST', '172.31.6.70'),
    'port': os.getenv('CLICKHOUSE_PORT', '8123'),
    'database': os.getenv('CLICKHOUSE_DATABASE', 'prod'),
    'buffer_size': os.getenv('CLICKHOUSE_BUFFER_SIZE', '50000'),
    'username': os.getenv('CLICKHOUSE_USERNAME', 'readonly_user'),
    'password': os.getenv('CLICKHOUSE_PASSWORD', 'cd267e1a-4db5-4936-a4bb-89f1104ea163')
}

# Database Tables
DATABASE_TABLES = {
    'channel_share': 'channel_share'  # Only table we actually query
}



def get_database_url() -> str:
    """Get the database URL from configuration"""
    return f"http://{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/?database={DATABASE_CONFIG['database']}&buffer_size={DATABASE_CONFIG['buffer_size']}"

def get_database_headers() -> dict:
    """Get authenticated headers for database requests"""
    import base64
    
    credentials = f"{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    return {
        'Content-Type': 'text/plain',
        'Accept': 'application/json',
        'Authorization': f'Basic {encoded_credentials}'
    }

# File Paths - Use robust path resolution
import os

# Get the project root directory (one level up from modules)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATHS = {
    'channel_share': 'database',  # Now using database instead of Excel file
    'grp_factors': os.path.join(PROJECT_ROOT, 'inputs', 'grp_based_reach_factor.csv'),
    'digital_incremental': os.path.join(PROJECT_ROOT, 'inputs', 'digital_incremental%.csv'),
    'base_population': os.path.join(PROJECT_ROOT, 'inputs', 'Base_population.csv')
}

# Default Regions - Will be dynamically loaded from database
# Fallback to India if database query fails
DEFAULT_REGIONS = ['India']

def get_available_regions(yearmonth: str = None) -> list:
    """
    Get available regions dynamically from database or use fallback
    
    Args:
        yearmonth (str): Date in format 'YYYY-MM-01' for database query
        
    Returns:
        list: Available regions from database or fallback list
    """
    if yearmonth:
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from channel_reach import get_available_regions_from_database
            regions = get_available_regions_from_database(yearmonth)
            if regions:
                return regions
        except Exception as e:
            print(f"Error getting regions from database: {e}")
    
    # Fallback to default regions
    return DEFAULT_REGIONS

def test_clickhouse_connection() -> dict:
    """Test ClickHouse connection using configured credentials"""
    try:
        import requests
        from requests.exceptions import RequestException
        
        url = get_database_url()
        headers = get_database_headers()
        
        # Simple test query
        test_query = "SELECT 1 as test"
        response = requests.post(url, headers=headers, data=test_query, timeout=10)
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'message': 'ClickHouse connection successful',
                'host': DATABASE_CONFIG['host'],
                'response_time': response.elapsed.total_seconds()
            }
        else:
            return {
                'status': 'error',
                'message': f'ClickHouse connection failed with status {response.status_code}',
                'host': DATABASE_CONFIG['host'],
                'response_time': 0
            }
            
    except RequestException as e:
        return {
            'status': 'error',
            'message': f'ClickHouse connection failed: {str(e)}',
            'host': DATABASE_CONFIG['host'],
            'response_time': 0
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Unexpected error: {str(e)}',
            'host': DATABASE_CONFIG['host'],
            'response_time': 0
        }

# Campaign Configuration
CAMPAIGN_CONFIG = {
    'default_days': 28,
    'max_computation_time': 300,  # 5 minutes
    'ots_target': 4.0,  # Opportunities To See target
    'universe_size_fallback': 88000000  # 88M fallback
}

# UI Configuration
UI_CONFIG = {
    'page_title': "Cross-Media Dashboard",
    'page_icon': "📊",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Chart Configuration
CHART_CONFIG = {
    'max_text_length': 20,
    'words_per_line': 2,
    'chart_height': 500
}

# Cache Configuration
CACHE_CONFIG = {
    'ltv_cache_key': 'ltv_cache',
    'date_range_cache_key': 'date_range_cache',
    'channel_share_cache_key': 'channel_share_cache'
}
