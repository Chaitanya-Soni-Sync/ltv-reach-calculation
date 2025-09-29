import pandas as pd
import numpy as np

def format_dataframe_for_display(df, reset_index=True, round_decimals=2):
    """
    Format dataframe for consistent display with proper number formatting and row numbering
    
    Args:
        df (pd.DataFrame): Input dataframe
        reset_index (bool): Whether to reset index to start from 1
        round_decimals (int): Number of decimal places for rounding
        
    Returns:
        pd.DataFrame: Formatted dataframe ready for display
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    formatted_df = df.copy()
    
    # Reset index to start from 1 if requested
    if reset_index:
        formatted_df = formatted_df.reset_index(drop=True)
        formatted_df.index = formatted_df.index + 1
        formatted_df.index.name = 'Row'
    
    # Round all numeric columns to specified decimal places
    for col in formatted_df.select_dtypes(include=[np.number]).columns:
        formatted_df[col] = formatted_df[col].round(round_decimals)
    
    return formatted_df

def format_numeric_columns(df, round_decimals=2):
    """
    Format only numeric columns with specified decimal places
    
    Args:
        df (pd.DataFrame): Input dataframe
        round_decimals (int): Number of decimal places for rounding
        
    Returns:
        pd.DataFrame: Dataframe with formatted numeric columns
    """
    if df.empty:
        return df
    
    formatted_df = df.copy()
    
    # Round all numeric columns to specified decimal places
    for col in formatted_df.select_dtypes(include=[np.number]).columns:
        formatted_df[col] = formatted_df[col].round(round_decimals)
    
    return formatted_df

def reset_row_numbers(df):
    """
    Reset row numbers to start from 1 and reassign after any sorting
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with reset row numbers starting from 1
    """
    if df.empty:
        return df
    
    formatted_df = df.copy()
    formatted_df = formatted_df.reset_index(drop=True)
    formatted_df.index = formatted_df.index + 1
    formatted_df.index.name = 'Row'
    
    return formatted_df
