"""
Data Cleaning Module
Provides functions to detect and handle data quality issues including
missing values, outliers, wrong data types, and duplicates.
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers_iqr(df, column):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Args:
        df (pd.DataFrame): The DataFrame
        column (str): Column name to check for outliers
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df))
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method.
    
    Args:
        df (pd.DataFrame): The DataFrame
        column (str): Column name to check for outliers
        threshold (float): Z-score threshold (default: 3)
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df))
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outlier_indices = df[column].dropna().index[z_scores > threshold]
    
    result = pd.Series([False] * len(df), index=df.index)
    result[outlier_indices] = True
    return result


def calculate_data_quality_score(df):
    """
    Calculate an overall data quality score (0-100).
    
    Args:
        df (pd.DataFrame): The DataFrame to evaluate
    
    Returns:
        dict: Dictionary containing score and breakdown
    """
    scores = {}
    
    # Completeness score (no missing values = 100)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    scores['completeness'] = completeness
    
    # Uniqueness score (no duplicates = 100)
    duplicate_rows = df.duplicated().sum()
    uniqueness = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 0
    scores['uniqueness'] = uniqueness
    
    # Consistency score (proper data types)
    consistency_issues = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            try:
                pd.to_numeric(df[col].dropna())
                consistency_issues += 1  # Could be numeric
            except:
                pass
    
    consistency = ((len(df.columns) - consistency_issues) / len(df.columns) * 100) if len(df.columns) > 0 else 0
    scores['consistency'] = consistency
    
    # Outlier score (fewer outliers = better)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        total_outliers = 0
        for col in numeric_cols:
            outliers = detect_outliers_iqr(df, col)
            total_outliers += outliers.sum()
        
        total_numeric_cells = len(df) * len(numeric_cols)
        outlier_score = ((total_numeric_cells - total_outliers) / total_numeric_cells * 100) if total_numeric_cells > 0 else 100
        scores['outliers'] = outlier_score
    else:
        scores['outliers'] = 100
    
    # Overall score (weighted average)
    overall_score = (
        scores['completeness'] * 0.35 +
        scores['uniqueness'] * 0.25 +
        scores['consistency'] * 0.20 +
        scores['outliers'] * 0.20
    )
    
    scores['overall'] = overall_score
    
    return scores


def get_data_quality_report(df):
    """
    Generate a comprehensive data quality report.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
    
    Returns:
        dict: Dictionary containing various data quality metrics
    """
    report = {}
    
    # Basic info
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    report['total_cells'] = len(df) * len(df.columns)
    
    # Missing values
    report['missing_values'] = df.isnull().sum().sum()
    report['missing_percentage'] = (report['missing_values'] / report['total_cells'] * 100) if report['total_cells'] > 0 else 0
    
    # Duplicates
    report['duplicate_rows'] = df.duplicated().sum()
    report['duplicate_percentage'] = (report['duplicate_rows'] / len(df) * 100) if len(df) > 0 else 0
    
    # Data types
    report['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
    report['categorical_columns'] = len(df.select_dtypes(include=['object', 'category']).columns)
    report['datetime_columns'] = len(df.select_dtypes(include=['datetime']).columns)
    
    # Column-level details
    column_details = []
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df) * 100) if len(df) > 0 else 0,
            'unique': df[col].nunique(),
            'unique_pct': (df[col].nunique() / len(df) * 100) if len(df) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['outliers_iqr'] = detect_outliers_iqr(df, col).sum()
        else:
            col_info['outliers_iqr'] = 0
        
        column_details.append(col_info)
    
    report['column_details'] = pd.DataFrame(column_details)
    report['quality_scores'] = calculate_data_quality_score(df)
    
    return report


def clean_data(df, options):
    """
    Clean the DataFrame based on selected options.
    
    Args:
        df (pd.DataFrame): The DataFrame to clean
        options (dict): Dictionary of cleaning options
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    cleaning_log = []
    
    # Handle missing values
    if options.get('handle_missing'):
        missing_strategy = options.get('missing_strategy', 'drop')
        
        if missing_strategy == 'drop_rows':
            before_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            removed = before_rows - len(df_cleaned)
            cleaning_log.append(f"Removed {removed} rows with missing values")
        
        elif missing_strategy == 'drop_columns':
            before_cols = len(df_cleaned.columns)
            threshold = options.get('missing_threshold', 0.5)
            df_cleaned = df_cleaned.dropna(axis=1, thresh=int(len(df_cleaned) * threshold))
            removed = before_cols - len(df_cleaned.columns)
            cleaning_log.append(f"Removed {removed} columns with >{(1-threshold)*100}% missing values")
        
        elif missing_strategy == 'fill_mean':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            cleaning_log.append(f"Filled missing values in numeric columns with mean")
        
        elif missing_strategy == 'fill_median':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            cleaning_log.append(f"Filled missing values in numeric columns with median")
        
        elif missing_strategy == 'fill_mode':
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col].fillna(mode_val[0], inplace=True)
            cleaning_log.append(f"Filled missing values with mode")
    
    # Remove duplicates
    if options.get('remove_duplicates'):
        before_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed = before_rows - len(df_cleaned)
        cleaning_log.append(f"Removed {removed} duplicate rows")
    
    # Handle outliers
    if options.get('handle_outliers'):
        outlier_method = options.get('outlier_method', 'iqr')
        outlier_strategy = options.get('outlier_strategy', 'remove')
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        for col in numeric_cols:
            if outlier_method == 'iqr':
                outliers = detect_outliers_iqr(df_cleaned, col)
            else:
                outliers = detect_outliers_zscore(df_cleaned, col)
            
            if outlier_strategy == 'remove':
                df_cleaned = df_cleaned[~outliers]
                total_outliers += outliers.sum()
            elif outlier_strategy == 'cap':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                total_outliers += outliers.sum()
        
        if outlier_strategy == 'remove':
            cleaning_log.append(f"Removed {total_outliers} outliers using {outlier_method.upper()} method")
        else:
            cleaning_log.append(f"Capped {total_outliers} outliers using {outlier_method.upper()} method")
    
    # Convert data types
    if options.get('convert_types'):
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col])
                    cleaning_log.append(f"Converted '{col}' to numeric type")
                except:
                    # Try to convert to datetime
                    try:
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                        cleaning_log.append(f"Converted '{col}' to datetime type")
                    except:
                        pass
    
    return df_cleaned, cleaning_log


def get_cleaning_comparison(df_before, df_after):
    """
    Generate a comparison report between original and cleaned data.
    
    Args:
        df_before (pd.DataFrame): Original DataFrame
        df_after (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    comparison = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'rows_removed': len(df_before) - len(df_after),
        'columns_before': len(df_before.columns),
        'columns_after': len(df_after.columns),
        'columns_removed': len(df_before.columns) - len(df_after.columns),
        'missing_before': df_before.isnull().sum().sum(),
        'missing_after': df_after.isnull().sum().sum(),
        'duplicates_before': df_before.duplicated().sum(),
        'duplicates_after': df_after.duplicated().sum(),
        'quality_before': calculate_data_quality_score(df_before),
        'quality_after': calculate_data_quality_score(df_after)
    }
    
    return comparison

