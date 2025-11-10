"""
Advanced EDA (Exploratory Data Analysis) Module
Provides functions for creating advanced visualizations including
correlation heatmaps, distribution plots, pairplots, and more.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64


def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): The DataFrame
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=800
    )
    
    return fig


def create_distribution_plots(df, columns=None, max_cols=6):
    """
    Create distribution plots for numeric columns.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to plot (if None, use all numeric)
        max_cols (int): Maximum number of columns to plot
    
    Returns:
        plotly.graph_objects.Figure: Distribution plots
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return None
    
    # Limit to max_cols
    numeric_cols = numeric_cols[:max_cols]
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig = make_subplots(
        rows=n_rows, 
        cols=min(3, n_cols),
        subplot_titles=numeric_cols
    )
    
    for idx, col in enumerate(numeric_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, nbinsx=30),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text='Distribution Plots',
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig


def create_box_plots_grid(df, columns=None, max_cols=6):
    """
    Create a grid of box plots for numeric columns.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to plot
        max_cols (int): Maximum number of columns to plot
    
    Returns:
        plotly.graph_objects.Figure: Box plots grid
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return None
    
    numeric_cols = numeric_cols[:max_cols]
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig = make_subplots(
        rows=n_rows,
        cols=min(3, n_cols),
        subplot_titles=numeric_cols
    )
    
    for idx, col in enumerate(numeric_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text='Box Plots - Outlier Detection',
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig


def create_pairplot_image(df, columns=None, max_cols=5):
    """
    Create a pairplot using seaborn and return as base64 image.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to include
        max_cols (int): Maximum number of columns
    
    Returns:
        str: Base64 encoded image
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols or len(numeric_cols) < 2:
        return None
    
    # Limit columns for performance
    numeric_cols = numeric_cols[:max_cols]
    
    # Sample data if too large
    sample_df = df[numeric_cols].sample(n=min(1000, len(df)))
    
    # Create pairplot
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(sample_df, diag_kind='hist', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle('Pairplot - Relationship Between Features', y=1.01)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_base64


def create_violin_plots(df, columns=None, max_cols=4):
    """
    Create violin plots for numeric columns.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to plot
        max_cols (int): Maximum number of columns
    
    Returns:
        plotly.graph_objects.Figure: Violin plots
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return None
    
    numeric_cols = numeric_cols[:max_cols]
    
    fig = go.Figure()
    
    for col in numeric_cols:
        fig.add_trace(go.Violin(
            y=df[col].dropna(),
            name=col,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title='Violin Plots - Distribution & Density',
        yaxis_title='Value',
        height=500,
        showlegend=True
    )
    
    return fig


def create_scatter_matrix(df, columns=None, max_cols=5):
    """
    Create a scatter plot matrix.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to include
        max_cols (int): Maximum number of columns
    
    Returns:
        plotly.graph_objects.Figure: Scatter matrix
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols or len(numeric_cols) < 2:
        return None
    
    numeric_cols = numeric_cols[:max_cols]
    
    # Sample if too large
    sample_df = df[numeric_cols].sample(n=min(1000, len(df)))
    
    fig = px.scatter_matrix(
        sample_df,
        dimensions=numeric_cols,
        title='Scatter Matrix - Feature Relationships'
    )
    
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=800, width=1000)
    
    return fig


def create_qq_plots(df, columns=None, max_cols=4):
    """
    Create Q-Q plots to check normality of distributions.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to plot
        max_cols (int): Maximum number of columns
    
    Returns:
        plotly.graph_objects.Figure: Q-Q plots
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return None
    
    numeric_cols = numeric_cols[:max_cols]
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 1) // 2
    
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=numeric_cols
    )
    
    for idx, col in enumerate(numeric_cols):
        row = idx // 2 + 1
        col_pos = idx % 2 + 1
        
        data = df[col].dropna()
        qq_data = stats.probplot(data, dist="norm")
        
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name=col,
                marker=dict(color='blue', size=5)
            ),
            row=row, col=col_pos
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title_text='Q-Q Plots - Normality Check',
        height=400 * n_rows,
        showlegend=False
    )
    
    return fig


def create_kde_plots(df, columns=None, max_cols=4):
    """
    Create Kernel Density Estimation plots.
    
    Args:
        df (pd.DataFrame): The DataFrame
        columns (list): List of columns to plot
        max_cols (int): Maximum number of columns
    
    Returns:
        plotly.graph_objects.Figure: KDE plots
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return None
    
    numeric_cols = numeric_cols[:max_cols]
    
    fig = go.Figure()
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Create KDE using histogram
        fig.add_trace(go.Histogram(
            x=data,
            name=col,
            histnorm='probability density',
            opacity=0.6
        ))
    
    fig.update_layout(
        title='Kernel Density Estimation',
        xaxis_title='Value',
        yaxis_title='Density',
        height=500,
        barmode='overlay'
    )
    
    return fig


# Import scipy.stats for qq plots
from scipy import stats

