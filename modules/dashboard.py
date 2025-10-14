#!/usr/bin/env python3
"""
LTV Reach Dashboard - Streamlit Application
Provides interactive UI for LTV reach analysis with filters and visualizations
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from main import compute_ltv_reach_pipeline
from dashboard_ui import render_filters, render_results_table
from dashboard_data import get_available_products, get_available_regions_list


def main():
    """Main dashboard application"""
    
    # Page configuration
    st.set_page_config(
        page_title="LTV Reach Calculator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📊 LTV Reach Calculator")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Get filter values
    filters = render_filters()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Analysis Parameters")
        st.write(f"**Product:** {filters['product']}")
        st.write(f"**Date Range:** {filters['start_date']} to {filters['end_date']}")
        st.write(f"**Regions:** {', '.join(filters['regions']) if filters['regions'] else 'All'}")
    
    with col2:
        st.subheader("Actions")
        run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Run analysis when button is clicked
    if run_button:
        with st.spinner("🔄 Running LTV reach analysis..."):
            try:
                # Run the pipeline
                results = compute_ltv_reach_pipeline(
                    products=filters['product'],
                    start_date=filters['start_date'],
                    end_date=filters['end_date'],
                    regions=filters['regions'] if filters['regions'] else None
                )
                
                # Check for errors
                if "error" in results:
                    st.error(f"❌ Error: {results['error']}")
                else:
                    # Store results in session state
                    st.session_state['results'] = results
                    st.success("✅ Analysis completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
    
    # Display results if available
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        if "error" not in results:
            # Display results table
            render_results_table(results)
            
            # Display summary metrics
            if 'daily_reach' in results and not results['daily_reach'].empty:
                daily_reach = results['daily_reach']
                
                st.markdown("---")
                st.subheader("📈 Summary Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Days", len(daily_reach))
                
                with col2:
                    st.metric("Max LTV Reach", f"{daily_reach['reach_final'].max():.2f}%")
                
                with col3:
                    st.metric("Avg Daily Reach", f"{daily_reach['reach_final'].mean():.2f}%")
                
                with col4:
                    total_spots = daily_reach['total_cumulative_spots'].max()
                    st.metric("Total Spots", f"{int(total_spots):,}")
            
            # Display LTV Reach with Rule No. 2
            if 'ltv_reach' in results and not results['ltv_reach'].empty:
                st.markdown("---")
                st.subheader("🎯 Final LTV Reach (with Rule No. 2)")
                
                ltv_reach = results['ltv_reach']
                
                # Format and display
                display_df = ltv_reach.rename(columns={
                    'brand': 'Brand',
                    'ltv_reach': 'LTV Reach (%)',
                    'original_reach': 'Original Reach (%)',
                    'region': 'Region',
                    'total_spots': 'Total Spots',
                    'total_rating': 'Total Rating',
                    'max_reach': 'Max Reach (%)',
                    'campaign_type': 'Campaign Type',
                    'rule_2_applied': 'Rule No. 2 Applied',
                    'date_range': 'Date Range'
                })
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Rule No. 2 Summary
                rule_2_applied = ltv_reach[ltv_reach['rule_2_applied'] == 'Yes']
                if not rule_2_applied.empty:
                    st.markdown("---")
                    st.subheader("📊 Rule No. 2 Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Campaigns with Rule No. 2", len(rule_2_applied))
                    
                    with col2:
                        avg_original = rule_2_applied['original_reach'].mean()
                        st.metric("Avg Original Reach", f"{avg_original:.2f}%")
                    
                    with col3:
                        avg_adjusted = rule_2_applied['ltv_reach'].mean()
                        avg_adjustment = avg_original - avg_adjusted
                        st.metric("Avg Adjustment", f"{avg_adjustment:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>LTV Reach Calculator v1.0 | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

