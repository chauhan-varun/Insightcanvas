"""
AI-Powered Data Visualization Dashboard
Main Streamlit application for uploading CSV files, visualizing data, 
and generating AI insights using the Groq API.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ai import generate_insights, chat_with_data
from data_cleaner import (
    get_data_quality_report, 
    calculate_data_quality_score,
    clean_data,
    get_cleaning_comparison
)
from advanced_eda import (
    create_correlation_heatmap,
    create_distribution_plots,
    create_box_plots_grid,
    create_pairplot_image,
    create_violin_plots,
    create_scatter_matrix,
    create_kde_plots
)


st.set_page_config(
    page_title="AI Data Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #262730;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application function"""
    
    st.markdown('<div class="main-header">📊 AI-Powered Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your CSV, visualize data, and get AI-powered insights</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file to analyze and visualize"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This dashboard allows you to:\n"
            "- Upload CSV files\n"
            "- Check data quality & get reports\n"
            "- Clean your data automatically\n"
            "- Create interactive visualizations\n"
            "- Advanced EDA with correlations\n"
            "- Get AI-powered insights\n"
            "- Chat with your data"
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.session_state['df'] = df
            
            st.success(f"✅ File uploaded successfully! Loaded {len(df)} rows and {len(df.columns)} columns.")
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📋 Data Preview", 
                "📊 Data Quality Report",
                "🧹 Data Cleaning",
                "📈 Visualizations", 
                "🔬 Advanced EDA",
                "🤖 AI Insights", 
                "💬 Chat with Data"
            ])
            
            with tab1:
                st.header("Data Preview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                st.subheader("First 10 Rows")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.subheader("Data Summary")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            with tab2:
                st.header("📊 Data Quality Report")
                st.write("Comprehensive analysis of your data quality including missing values, duplicates, and outliers.")
                
                quality_report = get_data_quality_report(df)
                quality_scores = quality_report['quality_scores']
                
                st.subheader("🎯 Overall Data Quality Score")
                overall_score = quality_scores['overall']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    score_color = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
                    st.metric("Overall Score", f"{overall_score:.1f}/100 {score_color}")
                
                with col2:
                    completeness = quality_scores['completeness']
                    score_color = "🟢" if completeness >= 90 else "🟡" if completeness >= 70 else "🔴"
                    st.metric("Completeness", f"{completeness:.1f}% {score_color}")
                
                with col3:
                    uniqueness = quality_scores['uniqueness']
                    score_color = "🟢" if uniqueness >= 95 else "🟡" if uniqueness >= 85 else "🔴"
                    st.metric("Uniqueness", f"{uniqueness:.1f}% {score_color}")
                
                with col4:
                    consistency = quality_scores['consistency']
                    score_color = "🟢" if consistency >= 90 else "🟡" if consistency >= 70 else "🔴"
                    st.metric("Consistency", f"{consistency:.1f}% {score_color}")
                
                with col5:
                    outlier_score = quality_scores['outliers']
                    score_color = "🟢" if outlier_score >= 90 else "🟡" if outlier_score >= 75 else "🔴"
                    st.metric("Outlier Score", f"{outlier_score:.1f}% {score_color}")
                
                st.markdown("---")
                
                st.subheader("📈 Basic Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", f"{quality_report['total_rows']:,}")
                
                with col2:
                    st.metric("Total Columns", quality_report['total_columns'])
                
                with col3:
                    st.metric("Missing Values", f"{quality_report['missing_values']:,} ({quality_report['missing_percentage']:.2f}%)")
                
                with col4:
                    st.metric("Duplicate Rows", f"{quality_report['duplicate_rows']:,} ({quality_report['duplicate_percentage']:.2f}%)")
                
                st.markdown("---")
                
                st.subheader("📊 Column Type Distribution")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Numeric Columns", quality_report['numeric_columns'])
                
                with col2:
                    st.metric("Categorical Columns", quality_report['categorical_columns'])
                
                with col3:
                    st.metric("DateTime Columns", quality_report['datetime_columns'])
                
                st.markdown("---")
                
                st.subheader("🔍 Detailed Column Analysis")
                column_details = quality_report['column_details']
                
                column_details_display = column_details.copy()
                column_details_display['missing_pct'] = column_details_display['missing_pct'].apply(lambda x: f"{x:.2f}%")
                column_details_display['unique_pct'] = column_details_display['unique_pct'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(column_details_display, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("💡 Recommendations")
                recommendations = []
                
                if quality_report['missing_percentage'] > 5:
                    recommendations.append("⚠️ **High missing values detected.** Consider using the Data Cleaning tab to handle missing values.")
                
                if quality_report['duplicate_rows'] > 0:
                    recommendations.append(f"⚠️ **Found {quality_report['duplicate_rows']} duplicate rows.** Consider removing them in the Data Cleaning tab.")
                
                if overall_score < 70:
                    recommendations.append("⚠️ **Low overall quality score.** Consider cleaning your data before analysis.")
                
                high_missing_cols = column_details[column_details['missing_pct'] > 50]['column'].tolist()
                if high_missing_cols:
                    recommendations.append(f"⚠️ **Columns with >50% missing values:** {', '.join(high_missing_cols)}")
                
                total_outliers = column_details['outliers_iqr'].sum()
                if total_outliers > 0:
                    recommendations.append(f"⚠️ **Detected {total_outliers} outliers** across numeric columns. Review in Advanced EDA tab.")
                
                if not recommendations:
                    st.success("✅ Your data quality looks good! No major issues detected.")
                else:
                    for rec in recommendations:
                        st.warning(rec)
            
            with tab3:
                st.header("🧹 Data Cleaning")
                st.write("Clean your data automatically based on detected issues.")
                
                quality_scores_before = calculate_data_quality_score(df)
                st.info(f"Current Data Quality Score: **{quality_scores_before['overall']:.1f}/100**")
                
                st.markdown("---")
                
                st.subheader("⚙️ Cleaning Options")
                
                with st.expander("🔧 Handle Missing Values", expanded=True):
                    handle_missing = st.checkbox("Handle missing values", value=True, key="handle_missing")
                    
                    if handle_missing:
                        missing_strategy = st.radio(
                            "Select strategy:",
                            ["drop_rows", "drop_columns", "fill_mean", "fill_median", "fill_mode"],
                            format_func=lambda x: {
                                "drop_rows": "Drop rows with missing values",
                                "drop_columns": "Drop columns with >50% missing values",
                                "fill_mean": "Fill numeric columns with mean",
                                "fill_median": "Fill numeric columns with median",
                                "fill_mode": "Fill all columns with mode"
                            }[x],
                            key="missing_strategy"
                        )
                        
                        if missing_strategy == "drop_columns":
                            missing_threshold = st.slider(
                                "Column missing value threshold",
                                min_value=0.1,
                                max_value=0.9,
                                value=0.5,
                                step=0.1,
                                help="Drop columns with missing values above this threshold",
                                key="missing_threshold"
                            )
                
                with st.expander("🔧 Handle Duplicates"):
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True, key="remove_duplicates")
                
                with st.expander("🔧 Handle Outliers"):
                    handle_outliers = st.checkbox("Handle outliers", value=False, key="handle_outliers")
                    
                    if handle_outliers:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            outlier_method = st.selectbox(
                                "Detection method:",
                                ["iqr", "zscore"],
                                format_func=lambda x: "IQR Method" if x == "iqr" else "Z-Score Method",
                                key="outlier_method"
                            )
                        
                        with col2:
                            outlier_strategy = st.selectbox(
                                "Handling strategy:",
                                ["remove", "cap"],
                                format_func=lambda x: "Remove outliers" if x == "remove" else "Cap outliers",
                                key="outlier_strategy"
                            )
                
                with st.expander("🔧 Auto-Convert Data Types"):
                    convert_types = st.checkbox(
                        "Automatically convert data types",
                        value=False,
                        help="Attempt to convert object columns to numeric or datetime",
                        key="convert_types"
                    )
                
                st.markdown("---")
                
                if st.button("🚀 Clean Data", type="primary", use_container_width=True):
                    with st.spinner("🧹 Cleaning data..."):
                        cleaning_options = {
                            'handle_missing': st.session_state.get('handle_missing', False),
                            'missing_strategy': st.session_state.get('missing_strategy', 'drop_rows'),
                            'missing_threshold': st.session_state.get('missing_threshold', 0.5),
                            'remove_duplicates': st.session_state.get('remove_duplicates', False),
                            'handle_outliers': st.session_state.get('handle_outliers', False),
                            'outlier_method': st.session_state.get('outlier_method', 'iqr'),
                            'outlier_strategy': st.session_state.get('outlier_strategy', 'remove'),
                            'convert_types': st.session_state.get('convert_types', False)
                        }
                        
                        df_cleaned, cleaning_log = clean_data(df, cleaning_options)
                        
                        st.session_state['df_cleaned'] = df_cleaned
                        st.session_state['cleaning_log'] = cleaning_log
                        st.session_state['df_original'] = df
                        
                        comparison = get_cleaning_comparison(df, df_cleaned)
                        st.session_state['comparison'] = comparison
                        
                        st.success("✅ Data cleaning completed!")
                
                if 'df_cleaned' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 Before vs After Comparison")
                    
                    comparison = st.session_state['comparison']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Rows",
                            f"{comparison['rows_after']:,}",
                            delta=f"{comparison['rows_removed']:,}" if comparison['rows_removed'] != 0 else "No change",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Missing Values",
                            f"{comparison['missing_after']:,}",
                            delta=f"{comparison['missing_before'] - comparison['missing_after']:,}",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Duplicates",
                            f"{comparison['duplicates_after']:,}",
                            delta=f"{comparison['duplicates_before'] - comparison['duplicates_after']:,}",
                            delta_color="inverse"
                        )
                    
                    st.subheader("🎯 Quality Score Improvement")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Before Cleaning**")
                        score_before = comparison['quality_before']['overall']
                        st.progress(score_before / 100)
                        st.write(f"Score: {score_before:.1f}/100")
                    
                    with col2:
                        st.markdown("**After Cleaning**")
                        score_after = comparison['quality_after']['overall']
                        st.progress(score_after / 100)
                        st.write(f"Score: {score_after:.1f}/100")
                    
                    improvement = score_after - score_before
                    if improvement > 0:
                        st.success(f"🎉 Quality improved by {improvement:.1f} points!")
                    elif improvement < 0:
                        st.warning(f"⚠️ Quality decreased by {abs(improvement):.1f} points")
                    else:
                        st.info("Quality score unchanged")
                    
                    st.subheader("📝 Cleaning Log")
                    for log_entry in st.session_state['cleaning_log']:
                        st.write(f"✓ {log_entry}")
                    
                    st.subheader("👀 Cleaned Data Preview")
                    st.dataframe(st.session_state['df_cleaned'].head(10), use_container_width=True)
                    
                    if st.button("✅ Use Cleaned Data for Analysis", type="primary"):
                        st.session_state['df'] = st.session_state['df_cleaned']
                        st.success("Cleaned data is now active! All visualizations and analyses will use the cleaned data.")
                        st.rerun()
            
            with tab4:
                st.header("Data Visualizations")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                all_cols = df.columns.tolist()
                
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns found for visualization.")
                else:
                    chart_type = st.selectbox(
                        "Select Chart Type",
                        ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    if chart_type == "Line Chart":
                        with col1:
                            x_col = st.selectbox("Select X-axis", all_cols, key="line_x")
                        with col2:
                            y_col = st.selectbox("Select Y-axis", numeric_cols, key="line_y")
                        
                        if st.button("Generate Line Chart", key="line_btn"):
                            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Bar Chart":
                        with col1:
                            x_col = st.selectbox("Select X-axis (Categories)", all_cols, key="bar_x")
                        with col2:
                            y_col = st.selectbox("Select Y-axis (Values)", numeric_cols, key="bar_y")
                        
                        if st.button("Generate Bar Chart", key="bar_btn"):
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Scatter Plot":
                        with col1:
                            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
                        with col2:
                            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
                        
                        color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="scatter_color")
                        
                        if st.button("Generate Scatter Plot", key="scatter_btn"):
                            if color_col == "None":
                                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                            else:
                                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Histogram":
                        with col1:
                            x_col = st.selectbox("Select Column", numeric_cols, key="hist_x")
                        with col2:
                            bins = st.slider("Number of Bins", 5, 100, 30)
                        
                        if st.button("Generate Histogram", key="hist_btn"):
                            fig = px.histogram(df, x=x_col, nbins=bins, title=f"Distribution of {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Box Plot":
                        with col1:
                            y_col = st.selectbox("Select Column", numeric_cols, key="box_y")
                        with col2:
                            x_col = st.selectbox("Group by (optional)", ["None"] + all_cols, key="box_x")
                        
                        if st.button("Generate Box Plot", key="box_btn"):
                            if x_col == "None":
                                fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
                            else:
                                fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Pie Chart":
                        with col1:
                            names_col = st.selectbox("Select Categories", all_cols, key="pie_names")
                        with col2:
                            values_col = st.selectbox("Select Values", numeric_cols, key="pie_values")
                        
                        if st.button("Generate Pie Chart", key="pie_btn"):
                            pie_data = df.groupby(names_col)[values_col].sum().reset_index()
                            fig = px.pie(pie_data, names=names_col, values=values_col, 
                                        title=f"{values_col} Distribution by {names_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.header("🔬 Advanced Exploratory Data Analysis")
                st.write("Deep dive into your data with correlation analysis, distribution plots, and more.")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns found for advanced EDA.")
                else:
                    st.subheader("🔥 Correlation Heatmap")
                    st.write("Understand relationships between numeric features.")
                    
                    if len(numeric_cols) >= 2:
                        with st.spinner("Creating correlation heatmap..."):
                            fig_corr = create_correlation_heatmap(df)
                            if fig_corr:
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                corr_matrix = df[numeric_cols].corr()
                                
                                corr_pairs = []
                                for i in range(len(corr_matrix.columns)):
                                    for j in range(i+1, len(corr_matrix.columns)):
                                        corr_pairs.append({
                                            'Feature 1': corr_matrix.columns[i],
                                            'Feature 2': corr_matrix.columns[j],
                                            'Correlation': corr_matrix.iloc[i, j]
                                        })
                                
                                if corr_pairs:
                                    corr_df = pd.DataFrame(corr_pairs)
                                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                                    
                                    st.markdown("**Top 5 Strongest Correlations:**")
                                    top_corr = corr_df.head(5)
                                    for idx, row in top_corr.iterrows():
                                        correlation = row['Correlation']
                                        color = "🔴" if abs(correlation) > 0.7 else "🟡" if abs(correlation) > 0.4 else "🟢"
                                        st.write(f"{color} **{row['Feature 1']}** ↔ **{row['Feature 2']}**: {correlation:.3f}")
                    else:
                        st.info("Need at least 2 numeric columns for correlation analysis.")
                    
                    st.markdown("---")
                    
                    st.subheader("📊 Distribution Analysis")
                    st.write("Visualize the distribution of your numeric features.")
                    
                    selected_dist_cols = st.multiselect(
                        "Select columns for distribution plots:",
                        numeric_cols,
                        default=numeric_cols[:min(6, len(numeric_cols))],
                        key="dist_cols"
                    )
                    
                    if selected_dist_cols:
                        with st.spinner("Creating distribution plots..."):
                            fig_dist = create_distribution_plots(df, selected_dist_cols)
                            if fig_dist:
                                st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("📦 Box Plots - Outlier Detection")
                    st.write("Identify outliers and understand data spread.")
                    
                    selected_box_cols = st.multiselect(
                        "Select columns for box plots:",
                        numeric_cols,
                        default=numeric_cols[:min(6, len(numeric_cols))],
                        key="box_cols"
                    )
                    
                    if selected_box_cols:
                        with st.spinner("Creating box plots..."):
                            fig_box = create_box_plots_grid(df, selected_box_cols)
                            if fig_box:
                                st.plotly_chart(fig_box, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("🎻 Violin Plots")
                    st.write("Combine box plots with density estimation.")
                    
                    selected_violin_cols = st.multiselect(
                        "Select columns for violin plots:",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key="violin_cols"
                    )
                    
                    if selected_violin_cols:
                        with st.spinner("Creating violin plots..."):
                            fig_violin = create_violin_plots(df, selected_violin_cols)
                            if fig_violin:
                                st.plotly_chart(fig_violin, use_container_width=True)
                    
                    st.markdown("---")
                    
                    if len(numeric_cols) >= 2:
                        st.subheader("🔗 Pairplot - Feature Relationships")
                        st.write("Visualize pairwise relationships between features.")
                        
                        selected_pair_cols = st.multiselect(
                            "Select columns for pairplot (max 5 for performance):",
                            numeric_cols,
                            default=numeric_cols[:min(4, len(numeric_cols))],
                            key="pair_cols"
                        )
                        
                        if len(selected_pair_cols) >= 2:
                            if st.button("Generate Pairplot", key="pairplot_btn"):
                                with st.spinner("Creating pairplot... (this may take a moment)"):
                                    img_base64 = create_pairplot_image(df, selected_pair_cols)
                                    if img_base64:
                                        st.image(f"data:image/png;base64,{img_base64}", use_column_width=True)
                                    else:
                                        st.error("Could not generate pairplot.")
                        else:
                            st.info("Select at least 2 columns to create a pairplot.")
                    
                    st.markdown("---")
                    
                    if len(numeric_cols) >= 2:
                        st.subheader("🎯 Interactive Scatter Matrix")
                        st.write("Interactive version of pairplot with zoom and pan.")
                        
                        selected_scatter_cols = st.multiselect(
                            "Select columns for scatter matrix:",
                            numeric_cols,
                            default=numeric_cols[:min(4, len(numeric_cols))],
                            key="scatter_matrix_cols"
                        )
                        
                        if len(selected_scatter_cols) >= 2:
                            if st.button("Generate Scatter Matrix", key="scatter_matrix_btn"):
                                with st.spinner("Creating scatter matrix..."):
                                    fig_scatter = create_scatter_matrix(df, selected_scatter_cols)
                                    if fig_scatter:
                                        st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Select at least 2 columns to create a scatter matrix.")
                    
                    st.markdown("---")
                    
                    st.subheader("🌊 Kernel Density Estimation")
                    st.write("Smooth distribution estimates for your data.")
                    
                    selected_kde_cols = st.multiselect(
                        "Select columns for KDE plots:",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key="kde_cols"
                    )
                    
                    if selected_kde_cols:
                        with st.spinner("Creating KDE plots..."):
                            fig_kde = create_kde_plots(df, selected_kde_cols)
                            if fig_kde:
                                st.plotly_chart(fig_kde, use_container_width=True)
            
            with tab6:
                st.header("🤖 AI-Powered Insights")
                st.write("Generate intelligent insights about your data using AI.")
                
                if st.button("🔮 Generate AI Insights", key="insights_btn", use_container_width=True):
                    with st.spinner("🤖 Analyzing your data with AI..."):
                        insights = generate_insights(df)
                        st.session_state['insights'] = insights
                
                if 'insights' in st.session_state:
                    st.markdown("### 📊 Analysis Results")
                    st.markdown(f'<div class="insight-box">{st.session_state["insights"]}</div>', 
                               unsafe_allow_html=True)
            
            with tab7:
                st.header("💬 Chat with Your Data")
                st.write("Ask questions about your data and get AI-powered answers.")
                
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
                
                for i, chat in enumerate(st.session_state['chat_history']):
                    with st.container():
                        st.markdown(f"**You:** {chat['question']}")
                        st.markdown(f"**AI:** {chat['answer']}")
                        st.markdown("---")
                
                with st.form(key='chat_form'):
                    user_question = st.text_input(
                        "Ask a question about your data:",
                        placeholder="e.g., What are the key trends in this dataset?"
                    )
                    submit_button = st.form_submit_button("Send")
                    
                    if submit_button and user_question:
                        with st.spinner("🤖 Thinking..."):
                            answer = chat_with_data(user_question, df)
                            
                            st.session_state['chat_history'].append({
                                'question': user_question,
                                'answer': answer
                            })
                            
                            st.rerun()
                
                if st.session_state['chat_history']:
                    if st.button("🗑️ Clear Chat History"):
                        st.session_state['chat_history'] = []
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted.")
    
    else:
        st.info("👈 Please upload a CSV file from the sidebar to get started!")
        
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload** your CSV file using the sidebar
        2. **Preview** your data in the Data Preview tab
        3. **Visualize** your data with interactive charts
        4. **Generate** AI-powered insights about your data
        5. **Chat** with your data to ask specific questions
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Make sure your CSV file has a header row with column names
        - Numeric columns work best for most visualizations
        - The AI can help you understand patterns and trends in your data
        - Try asking specific questions in the chat feature
        """)


if __name__ == "__main__":
    main()


