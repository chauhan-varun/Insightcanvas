"""
AI Module - Groq API Integration
This module provides helper functions to interact with the Groq API
for generating insights and chatting with data.
"""

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def get_groq_client():
    """
    Initialize and return a Groq client instance.
    
    Returns:
        Groq: Initialized Groq client
    
    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return Groq(api_key=api_key)


def dataframe_to_summary(df):
    """
    Convert a pandas DataFrame to a text summary for the AI model.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize
    
    Returns:
        str: A text summary of the DataFrame
    """
    summary = f"Dataset Overview:\n"
    summary += f"- Number of rows: {len(df)}\n"
    summary += f"- Number of columns: {len(df.columns)}\n"
    summary += f"- Column names: {', '.join(df.columns.tolist())}\n\n"
    
    summary += "Column Statistics:\n"
    categorical_cols = []
    numeric_cols = []
    
    for col in df.columns:
        summary += f"\n{col}:\n"
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            summary += f"  - Type: Numeric\n"
            summary += f"  - Min: {df[col].min()}\n"
            summary += f"  - Max: {df[col].max()}\n"
            summary += f"  - Mean: {df[col].mean():.2f}\n"
            summary += f"  - Median: {df[col].median():.2f}\n"
        else:
            categorical_cols.append(col)
            summary += f"  - Type: {df[col].dtype}\n"
            summary += f"  - Unique values: {df[col].nunique()}\n"
            if df[col].nunique() <= 30:
                summary += f"  - Values: {', '.join(map(str, df[col].unique().tolist()[:30]))}\n"
    
    # Add aggregated data for better insights
    summary += "\n" + "="*50 + "\n"
    summary += "AGGREGATED DATA FOR ANALYSIS:\n"
    summary += "="*50 + "\n"
    
    # For each categorical column, show aggregated numeric data
    for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[cat_col].nunique() <= 50:  # Only for columns with reasonable unique values
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    agg_data = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).round(2)
                    summary += f"\n{num_col} by {cat_col}:\n"
                    summary += agg_data.to_string() + "\n"
                except:
                    pass
    
    # Add sample rows
    summary += f"\n{'='*50}\n"
    summary += f"SAMPLE DATA (first 10 rows):\n"
    summary += f"{'='*50}\n"
    summary += df.head(10).to_string() + "\n"
    
    # Add sample of last rows too
    if len(df) > 10:
        summary += f"\nLast 5 rows:\n"
        summary += df.tail(5).to_string() + "\n"
    
    return summary


def generate_insights(dataframe):
    """
    Generate AI insights based on the uploaded DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze
    
    Returns:
        str: AI-generated insights about the data
    """
    try:
        client = get_groq_client()
        
        # Convert DataFrame to a summary string
        data_summary = dataframe_to_summary(dataframe)
        
        # Create the prompt for the AI model
        prompt = f"""You are a data analyst. Analyze the following dataset and provide meaningful insights:

{data_summary}

Please provide:
1. Key observations about the data
2. Interesting patterns or trends you notice
3. Potential correlations between variables
4. Suggestions for further analysis
5. Any data quality issues you notice

Keep your response clear, concise, and actionable."""
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data analyst who provides clear, actionable insights from data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1500
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"


def chat_with_data(question, dataframe):
    """
    Answer user questions about the data using the Groq API.
    
    Args:
        question (str): The user's question about the data
        dataframe (pd.DataFrame): The DataFrame to query
    
    Returns:
        str: AI-generated answer to the question
    """
    try:
        client = get_groq_client()
        
        # Convert DataFrame to a summary string
        data_summary = dataframe_to_summary(dataframe)
        
        # Create the prompt for the AI model
        prompt = f"""You have access to the following dataset:

{data_summary}

User Question: {question}

Please answer the question based on the data provided. Be specific and reference actual values from the data when possible."""
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data assistant. Answer questions about the provided dataset accurately and concisely."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1000
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error answering question: {str(e)}"

