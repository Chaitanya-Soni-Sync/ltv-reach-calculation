"""
Dashboard UI Components
Handles all UI rendering for the Streamlit dashboard
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dashboard_data import get_available_products, get_available_regions_list


def render_filters():
    """Render filter components in sidebar and return selected values"""
    
    # Product filter
    products = get_available_products()
    selected_product = st.sidebar.selectbox(
        "Product",
        options=products,
        index=0,
        help="Select the product to analyze"
    )
    
    # Date range filters
    st.sidebar.subheader("📅 Date Range")
    
    # Default dates
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - timedelta(days=30)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start_date,
            help="Select the start date for analysis"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end_date,
            help="Select the end date for analysis"
        )
    
    # Region filter
    st.sidebar.subheader("🌍 Regions")
    
    regions = get_available_regions_list()
    
    # Add "All Regions" option
    region_options = ["All Regions"] + regions
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=region_options,
        default=["All Regions"],
        help="Select one or more regions to analyze. Select 'All Regions' for complete analysis."
    )
    
    # Process region selection
    if "All Regions" in selected_regions or not selected_regions:
        final_regions = None  # None means all regions
    else:
        final_regions = selected_regions
    
    # Advanced options (collapsible)
    with st.sidebar.expander("⚙️ Advanced Options"):
        show_debug = st.checkbox("Show Debug Information", value=False)
        download_csv = st.checkbox("Enable CSV Download", value=True)
    
    return {
        'product': selected_product,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'regions': final_regions,
        'show_debug': show_debug,
        'download_csv': download_csv
    }


def render_results_table(results):
    """Render the main results table in specified format"""
    
    if 'daily_reach' not in results or results['daily_reach'].empty:
        st.warning("⚠️ No results to display")
        return
    
    daily_reach = results['daily_reach']
    
    # Prepare display DataFrame in specified format
    display_df = daily_reach.rename(columns={
        'region': 'Region',
        'program date': 'Date',
        'brand': 'Product',
        'total_cumulative_spots': 'Cumm_Spots',
        'max_reach': 'Max Reach',
        'total_rating': 'Rating',
        'reach_final': 'LTV_Reach'
    })
    
    # Format Date column
    display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Select and order columns as specified
    columns_order = ['Region', 'Date', 'Product', 'Cumm_Spots', 'Max Reach', 'Rating', 'LTV_Reach']
    display_df = display_df[columns_order]
    
    # Format numeric columns
    display_df['Cumm_Spots'] = display_df['Cumm_Spots'].astype(int)
    display_df['Max Reach'] = display_df['Max Reach'].round(2)
    display_df['Rating'] = display_df['Rating'].round(2)
    display_df['LTV_Reach'] = display_df['LTV_Reach'].round(2)
    
    # Display table
    st.subheader("📊 Daily LTV Reach Results")
    
    # Add search/filter options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search table", "", placeholder="Search by region, product, etc.")
    
    with col2:
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100, "All"], index=3)
    
    # Apply search filter
    if search_term:
        mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = display_df[mask]
    else:
        filtered_df = display_df
    
    # Display table with pagination or all rows
    if rows_per_page == "All":
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            height=600
        )
    else:
        st.dataframe(
            filtered_df.head(rows_per_page),
            use_container_width=True,
            hide_index=True
        )
    
    # Download button
    if st.session_state.get('results', {}).get('download_csv', True):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"ltv_reach_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Show record count
    st.caption(f"Showing {len(filtered_df)} of {len(display_df)} records")


def render_error(error_message):
    """Render error message"""
    st.error(f"❌ {error_message}")


def render_success(message):
    """Render success message"""
    st.success(f"✅ {message}")

