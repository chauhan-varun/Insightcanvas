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


# Page configuration
st.set_page_config(
    page_title="AI Data Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
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
    
    # Header
    st.markdown('<div class="main-header">📊 AI-Powered Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your CSV, visualize data, and get AI-powered insights</div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
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
            "- Preview and explore data\n"
            "- Create interactive visualizations\n"
            "- Get AI-powered insights\n"
            "- Chat with your data"
        )
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Store dataframe in session state
            st.session_state['df'] = df
            
            # Display success message
            st.success(f"✅ File uploaded successfully! Loaded {len(df)} rows and {len(df.columns)} columns.")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Visualizations", "🤖 AI Insights", "💬 Chat with Data"])
            
            # Tab 1: Data Preview
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
            
            # Tab 2: Visualizations
            with tab2:
                st.header("Data Visualizations")
                
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                all_cols = df.columns.tolist()
                
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns found for visualization.")
                else:
                    # Chart type selection
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
                            # Aggregate data if needed
                            pie_data = df.groupby(names_col)[values_col].sum().reset_index()
                            fig = px.pie(pie_data, names=names_col, values=values_col, 
                                        title=f"{values_col} Distribution by {names_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
            
            # Tab 3: AI Insights
            with tab3:
                st.header("🤖 AI-Powered Insights")
                st.write("Generate intelligent insights about your data using AI.")
                
                if st.button("🔮 Generate AI Insights", key="insights_btn", use_container_width=True):
                    with st.spinner("🤖 Analyzing your data with AI..."):
                        insights = generate_insights(df)
                        st.session_state['insights'] = insights
                
                # Display insights if available
                if 'insights' in st.session_state:
                    st.markdown("### 📊 Analysis Results")
                    st.markdown(f'<div class="insight-box">{st.session_state["insights"]}</div>', 
                               unsafe_allow_html=True)
            
            # Tab 4: Chat with Data
            with tab4:
                st.header("💬 Chat with Your Data")
                st.write("Ask questions about your data and get AI-powered answers.")
                
                # Initialize chat history
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
                
                # Display chat history
                for i, chat in enumerate(st.session_state['chat_history']):
                    with st.container():
                        st.markdown(f"**You:** {chat['question']}")
                        st.markdown(f"**AI:** {chat['answer']}")
                        st.markdown("---")
                
                # Chat input
                with st.form(key='chat_form'):
                    user_question = st.text_input(
                        "Ask a question about your data:",
                        placeholder="e.g., What are the key trends in this dataset?"
                    )
                    submit_button = st.form_submit_button("Send")
                    
                    if submit_button and user_question:
                        with st.spinner("🤖 Thinking..."):
                            answer = chat_with_data(user_question, df)
                            
                            # Add to chat history
                            st.session_state['chat_history'].append({
                                'question': user_question,
                                'answer': answer
                            })
                            
                            # Rerun to display new message
                            st.rerun()
                
                # Clear chat history button
                if st.session_state['chat_history']:
                    if st.button("🗑️ Clear Chat History"):
                        st.session_state['chat_history'] = []
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted.")
    
    else:
        # Landing page when no file is uploaded
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


