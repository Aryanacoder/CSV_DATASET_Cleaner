import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import chardet
import base64
import io
import re
import os
import time
import sys
import gc
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

# Import custom modules
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_visualizer import DataVisualizer
from lightweight_command_processor import LightweightCommandProcessor
from data_optimizer import DataOptimizer

# Set page configuration
st.set_page_config(
    page_title="AI CSV Cleaner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/csv-cleaner',
        'Report a bug': 'https://github.com/yourusername/csv-cleaner/issues',
        'About': 'AI-Powered CSV Dataset Cleaning Assistant with voice and text command capabilities.'
    }
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(to right, #4b6cb7, #182848);
        border-radius: 5px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4b6cb7;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    
    .command-suggestion {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .command-suggestion:hover {
        background-color: #cce5ff;
    }
    
    .stProgress .st-bo {
        background-color: #4b6cb7;
    }
    
    .stDataFrame {
        border-radius: 5px;
        overflow: hidden;
    }
    
    .dataframe-container {
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .voice-button {
        background-color: #4b6cb7;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .voice-button:hover {
        background-color: #3a5795;
    }
    
    .voice-button:active {
        background-color: #2a4374;
    }
    
    .command-history {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    .command-item {
        padding: 0.25rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .command-item:last-child {
        border-bottom: none;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4b6cb7;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    .tab-content {
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0 5px 5px 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize lightweight command processor
@st.cache_resource
def load_command_processor():
    try:
        # Initialize lightweight command processor
        command_processor = LightweightCommandProcessor()
        return command_processor
    except Exception as e:
        st.error(f"Error initializing command processor: {e}")
        return None

# Function to get download link for dataframe
def get_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Function to create a placeholder for voice recording
def create_voice_placeholder():
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
            <button class="voice-button" id="recordButton">
                <i class="fas fa-microphone"></i> Start Voice Command
            </button>
        </div>
        <div id="recordingStatus" style="text-align: center; margin-top: 10px; color: #666;">
            Click to start recording
        </div>
        """, unsafe_allow_html=True)
    return placeholder

# Function to display metrics
def display_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Rows</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Columns</div>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col3:
        missing_values = df.isnull().sum().sum()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """.format(missing_values), unsafe_allow_html=True)
    
    with col4:
        duplicates = df.duplicated().sum()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Duplicates</div>
        </div>
        """.format(duplicates), unsafe_allow_html=True)

# Function to display data quality report
def display_data_quality(df):
    st.subheader("Data Quality Report")
    
    # Data types
    st.markdown("### Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': [str(df[col].dtype) for col in df.columns],
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing Values': [df[col].isnull().sum() for col in df.columns],
        'Missing (%)': [(df[col].isnull().sum() / len(df) * 100).round(2) for col in df.columns]
    })
    st.dataframe(dtypes_df, use_container_width=True)
    
    # Missing values visualization
    st.markdown("### Missing Values")
    if df.isnull().sum().sum() > 0:
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': [df[col].isnull().sum() for col in df.columns],
            'Missing (%)': [(df[col].isnull().sum() / len(df) * 100).round(2) for col in df.columns]
        }).sort_values('Missing Count', ascending=False)
        
        fig = px.bar(
            missing_df[missing_df['Missing Count'] > 0], 
            x='Column', 
            y='Missing (%)',
            title='Missing Values by Column (%)',
            color='Missing (%)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values found in the dataset.")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.markdown("### Numeric Columns Statistics")
        stats_df = df[numeric_cols].describe().T
        stats_df['skew'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurtosis()
        st.dataframe(stats_df, use_container_width=True)
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.markdown("### Categorical Columns Statistics")
        cat_stats = []
        for col in categorical_cols:
            cat_stats.append({
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Most Common': df[col].value_counts().index[0] if not df[col].value_counts().empty else None,
                'Frequency': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
                'Frequency (%)': (df[col].value_counts().iloc[0] / len(df) * 100).round(2) if not df[col].value_counts().empty else 0
            })
        cat_stats_df = pd.DataFrame(cat_stats)
        st.dataframe(cat_stats_df, use_container_width=True)

# Function to display data insights
def display_data_insights(df, command_processor):
    st.subheader("Data Insights")
    
    try:
        # Generate insights using the command processor
        insights = command_processor.generate_insights(df)
        
        # Display basic statistics
        st.markdown("### Dataset Overview")
        st.markdown(f"**Rows:** {insights['basic_stats']['rows']} | **Columns:** {insights['basic_stats']['columns']}")
        
        # Display missing values insights
        st.markdown("### Missing Values")
        total_missing = insights['missing_values']['total']
        if total_missing > 0:
            st.markdown(f"**Total Missing Values:** {total_missing}")
            missing_cols = [col for col, count in insights['missing_values']['by_column'].items() if count > 0]
            if missing_cols:
                st.markdown(f"**Columns with Missing Values:** {', '.join(missing_cols)}")
        else:
            st.markdown("No missing values found in the dataset.")
        
        # Display duplicates insights
        st.markdown("### Duplicates")
        duplicates = insights['duplicates']['count']
        if duplicates > 0:
            st.markdown(f"**Duplicate Rows:** {duplicates} ({(duplicates / insights['basic_stats']['rows'] * 100):.2f}%)")
        else:
            st.markdown("No duplicate rows found in the dataset.")
        
        # Display data types insights
        st.markdown("### Data Types")
        st.markdown(f"**Numeric Columns:** {len(insights['data_types']['numeric'])}")
        st.markdown(f"**Categorical Columns:** {len(insights['data_types']['categorical'])}")
        st.markdown(f"**Datetime Columns:** {len(insights['data_types']['datetime'])}")
        
        # If using HuggingFace integration, display additional insights
        if hasattr(command_processor, 'use_hf') and command_processor.use_hf:
            try:
                hf_insights = command_processor.hf_integration.generate_data_insights(df)
                
                # Display correlations
                if 'correlations' in hf_insights and 'top_pairs' in hf_insights['correlations']:
                    st.markdown("### Top Correlations")
                    for col1, col2, corr_value in hf_insights['correlations']['top_pairs']:
                        st.markdown(f"**{col1}** and **{col2}**: {corr_value:.3f}")
                
                # Display outliers
                if 'outliers' in hf_insights and hf_insights['outliers']:
                    st.markdown("### Outliers")
                    for col, info in hf_insights['outliers'].items():
                        st.markdown(f"**{col}**: {info['count']} outliers ({info['percent']}%)")
                
                # Display recommendations
                if 'recommendations' in hf_insights and hf_insights['recommendations']:
                    st.markdown("### Recommendations")
                    for i, rec in enumerate(hf_insights['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
            except Exception as e:
                st.warning(f"Could not generate advanced insights: {e}")
    
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        st.markdown("### Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# Function to display cleaning history
def display_cleaning_history(history):
    st.subheader("Cleaning History")
    
    if not history:
        st.info("No cleaning operations performed yet.")
        return
    
    for i, entry in enumerate(history, 1):
        with st.expander(f"{i}. {entry['operation']} - {entry['timestamp']}"):
            st.markdown(f"**Command:** {entry['command']}")
            st.markdown(f"**Result:** {entry['result']}")
            if 'details' in entry and entry['details']:
                st.markdown("**Details:**")
                for key, value in entry['details'].items():
                    st.markdown(f"- {key}: {value}")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'data_cleaner' not in st.session_state:
    st.session_state.data_cleaner = DataCleaner()
if 'data_visualizer' not in st.session_state:
    st.session_state.data_visualizer = DataVisualizer()
if 'data_optimizer' not in st.session_state:
    st.session_state.data_optimizer = DataOptimizer()
if 'command_processor' not in st.session_state:
    st.session_state.command_processor = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'command_history' not in st.session_state:
    st.session_state.command_history = []
if 'file_details' not in st.session_state:
    st.session_state.file_details = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Data"
if 'visualization_result' not in st.session_state:
    st.session_state.visualization_result = None
if 'data_insights' not in st.session_state:
    st.session_state.data_insights = None

# Custom header
st.markdown("<h1 class='main-header'>AI-Powered CSV Dataset Cleaning Assistant</h1>", unsafe_allow_html=True)
st.markdown("This application helps you clean, preprocess, and visualize CSV datasets using voice or text commands. Upload your CSV file, then use the voice or text input to issue cleaning commands.")

# Load command processor if not already loaded
if st.session_state.command_processor is None:
    with st.spinner("Initializing command processor..."):
        st.session_state.command_processor = load_command_processor()
    
# Sidebar for data upload and voice commands
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Dataset Upload</h2>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Check if this is a new file
            if st.session_state.file_details is None or st.session_state.file_details['name'] != uploaded_file.name:
                with st.spinner("Loading and analyzing dataset..."):
                    # Detect encoding and load data
                    content = uploaded_file.read()
                    encoding_result = chardet.detect(content)
                    encoding = encoding_result['encoding']
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Load data with detected encoding
                    df, load_info = st.session_state.data_loader.load_csv(
                        uploaded_file, 
                        encoding=encoding,
                        optimize=True
                    )
                    
                    # Store data in session state
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    
                    # Store file details
                    st.session_state.file_details = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'encoding': encoding,
                        'confidence': encoding_result['confidence'],
                        'load_info': load_info
                    }
                    
                    # Reset history
                    st.session_state.cleaning_history = []
                    st.session_state.command_history = []
                    st.session_state.visualization_result = None
                    
                    # Generate data insights
                    st.session_state.data_insights = st.session_state.command_processor.generate_insights(df)
                
                st.success(f"Successfully loaded {uploaded_file.name}")
                st.markdown(f"**Encoding:** {encoding} (Confidence: {encoding_result['confidence']:.2f})")
                st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
                
                # Display memory usage
                memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                st.markdown(f"**Memory Usage:** {memory_usage:.2f} MB")
                
                # Display optimization info if available
                if 'memory_reduction' in load_info:
                    st.markdown(f"**Memory Reduction:** {load_info['memory_reduction']:.2f}%")
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    # Command input section
    st.markdown("<h2 class='sub-header'>Command Input</h2>", unsafe_allow_html=True)
    
    # Voice command section
    st.markdown("<h3>Voice Command</h3>", unsafe_allow_html=True)
    voice_button = st.button("🎤 Start Voice Command")
    
    # Text command section
    st.markdown("<h3>Text Command</h3>", unsafe_allow_html=True)
    text_command = st.text_input("Enter your command:")
    submit_button = st.button("Submit Command")
    
    # Command suggestions
    if st.session_state.data is not None:
        st.markdown("<h3>Command Suggestions</h3>", unsafe_allow_html=True)
        
        # Generate suggestions based on dataset
        suggestions = st.session_state.command_processor.generate_suggestions(st.session_state.data)
        
        # Display suggestions as clickable buttons
        for suggestion in suggestions:
            if st.button(suggestion):
                # Set the suggestion as the command and process it
                text_command = suggestion
                submit_button = True
    
    # Reset button
    if st.session_state.data is not None and st.session_state.original_data is not None:
        if st.button("Reset to Original Data"):
            st.session_state.data = st.session_state.original_data.copy()
            st.session_state.cleaning_history = []
            st.session_state.visualization_result = None
            st.success("Reset to original data")

# Main application layout
def main():
    # Load command processor if not already loaded
    if st.session_state.command_processor is None:
        with st.spinner("Initializing command processor..."):
            st.session_state.command_processor = load_command_processor()
    
    # Sidebar for data upload and voice commands
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Dataset Upload</h2>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Check if this is a new file
                if st.session_state.file_details is None or st.session_state.file_details['name'] != uploaded_file.name:
                    with st.spinner("Loading and analyzing dataset..."):
                        # Detect encoding and load data
                        content = uploaded_file.read()
                        encoding_result = chardet.detect(content)
                        encoding = encoding_result['encoding']
                        
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Load data with detected encoding
                        df, load_info = st.session_state.data_loader.load_csv(
                            uploaded_file, 
                            encoding=encoding,
                            optimize=True
                        )
                        
                        # Store data in session state
                        st.session_state.data = df
                        st.session_state.original_data = df.copy()
                        
                        # Store file details
                        st.session_state.file_details = {
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'encoding': encoding,
                            'confidence': encoding_result['confidence'],
                            'load_info': load_info
                        }
                        
                        # Reset history
                        st.session_state.cleaning_history = []
                        st.session_state.command_history = []
                        st.session_state.visualization_result = None
                        
                        # Generate data insights
                        st.session_state.data_insights = st.session_state.command_processor.generate_insights(df)
                    
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    st.markdown(f"**Encoding:** {encoding} (Confidence: {encoding_result['confidence']:.2f})")
                    st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
                    
                    # Display memory usage
                    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
                    st.markdown(f"**Memory Usage:** {memory_usage:.2f} MB")
                    
                    # Display optimization info if available
                    if 'memory_reduction' in load_info:
                        st.markdown(f"**Memory Reduction:** {load_info['memory_reduction']:.2f}%")
            
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        
        # Command input section
        st.markdown("<h2 class='sub-header'>Command Input</h2>", unsafe_allow_html=True)
        
        # Voice command section
        st.markdown("<h3>Voice Command</h3>", unsafe_allow_html=True)
        voice_button = st.button("🎤 Start Voice Command")
        
        # Text command section
        st.markdown("<h3>Text Command</h3>", unsafe_allow_html=True)
        text_command = st.text_input("Enter your command:")
        submit_button = st.button("Submit Command")
        
        # Command suggestions
        if st.session_state.data is not None:
            st.markdown("<h3>Command Suggestions</h3>", unsafe_allow_html=True)
            
            # Generate suggestions based on dataset
            suggestions = st.session_state.command_processor.generate_suggestions(st.session_state.data)
            
            # Display suggestions as clickable buttons
            for suggestion in suggestions:
                if st.button(suggestion):
                    # Set the suggestion as the command and process it
                    text_command = suggestion
                    submit_button = True
        
        # Reset button
        if st.session_state.data is not None and st.session_state.original_data is not None:
            if st.button("Reset to Original Data"):
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.cleaning_history = []
                st.session_state.visualization_result = None
                st.success("Reset to original data")
    
    # Main content area
    if st.session_state.data is not None:
        # Display tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Statistics", "Visualizations", "Cleaning History", "Export"])
        
        # Data tab
        with tab1:
            st.subheader("Dataset Preview")
            
            # Display metrics
            display_metrics(st.session_state.data)
            
            # Display dataframe
            st.dataframe(st.session_state.data, use_container_width=True)
            
            # Display column information
            with st.expander("Column Information"):
                col_info = pd.DataFrame({
                    'Column': st.session_state.data.columns,
                    'Data Type': st.session_state.data.dtypes.values,
                    'Non-Null Count': st.session_state.data.count().values,
                    'Null Count': st.session_state.data.isnull().sum().values,
                    'Unique Values': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
                })
                st.dataframe(col_info, use_container_width=True)
        
        # Statistics tab
        with tab2:
            # Display data quality report
            display_data_quality(st.session_state.data)
            
            # Display data insights
            display_data_insights(st.session_state.data, st.session_state.command_processor)
        
        # Visualizations tab
        with tab3:
            st.subheader("Data Visualizations")
            
            # Display current visualization if available
            if st.session_state.visualization_result is not None:
                if 'fig' in st.session_state.visualization_result:
                    st.plotly_chart(st.session_state.visualization_result['fig'], use_container_width=True)
                elif 'error' in st.session_state.visualization_result:
                    st.error(st.session_state.visualization_result['error'])
            else:
                st.info("Use commands like 'Show correlation heatmap' or 'Plot histogram of [column]' to generate visualizations.")
            
            # Quick visualization options
            st.markdown("### Quick Visualizations")
            
            # Correlation heatmap
            if st.button("Generate Correlation Heatmap"):
                with st.spinner("Generating correlation heatmap..."):
                    viz_result = st.session_state.data_visualizer.create_correlation_heatmap(st.session_state.data)
                    st.session_state.visualization_result = viz_result
                    st.experimental_rerun()
            
            # Histograms for numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select column for histogram:", numeric_cols)
                
                with col2:
                    if st.button("Generate Histogram"):
                        with st.spinner(f"Generating histogram for {selected_col}..."):
                            viz_result = st.session_state.data_visualizer.create_histogram(
                                st.session_state.data, 
                                column=selected_col
                            )
                            st.session_state.visualization_result = viz_result
                            st.experimental_rerun()
            
            # Scatter plot for numeric columns
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("Select X column:", numeric_cols)
                
                with col2:
                    y_col = st.selectbox("Select Y column:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                with col3:
                    if st.button("Generate Scatter Plot"):
                        with st.spinner(f"Generating scatter plot for {y_col} vs {x_col}..."):
                            viz_result = st.session_state.data_visualizer.create_scatter_plot(
                                st.session_state.data, 
                                x_column=x_col,
                                y_column=y_col
                            )
                            st.session_state.visualization_result = viz_result
                            st.experimental_rerun()
            
            # Missing values visualization
            if st.button("Visualize Missing Values"):
                with st.spinner("Generating missing values visualization..."):
                    viz_result = st.session_state.data_visualizer.create_missing_values_chart(st.session_state.data)
                    st.session_state.visualization_result = viz_result
                    st.experimental_rerun()
        
        # Cleaning History tab
        with tab4:
            display_cleaning_history(st.session_state.cleaning_history)
        
        # Export tab
        with tab5:
            st.subheader("Export Cleaned Dataset")
            
            # Download options
            st.markdown("### Download Options")
            
            # CSV download
            if st.button("Download as CSV"):
                csv = st.session_state.data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Excel download
            if st.button("Download as Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.data.to_excel(writer, index=False, sheet_name='Cleaned Data')
                    # Add a sheet with cleaning history
                    if st.session_state.cleaning_history:
                        history_df = pd.DataFrame([
                            {
                                'Timestamp': entry['timestamp'],
                                'Operation': entry['operation'],
                                'Command': entry['command'],
                                'Result': entry['result']
                            }
                            for entry in st.session_state.cleaning_history
                        ])
                        history_df.to_excel(writer, index=False, sheet_name='Cleaning History')
                
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cleaned_data.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # JSON download
            if st.button("Download as JSON"):
                json_str = st.session_state.data.to_json(orient='records')
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="cleaned_data.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Download with cleaning history
            if st.session_state.cleaning_history:
                st.markdown("### Download with Cleaning History")
                
                if st.button("Download Data with Cleaning History"):
                    # Create a dictionary with data and cleaning history
                    export_data = {
                        'data': st.session_state.data.to_dict(orient='records'),
                        'cleaning_history': st.session_state.cleaning_history,
                        'original_shape': st.session_state.original_data.shape,
                        'cleaned_shape': st.session_state.data.shape,
                        'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Convert to JSON
                    json_str = json.dumps(export_data, default=str)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="data_with_history.json">Download Data with History</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    else:
        # Display welcome message if no data is loaded
        st.info("👈 Please upload a CSV file to get started.")
        
        # Display sample datasets
        st.markdown("### Sample Datasets")
        st.markdown("Don't have a dataset? Try one of these samples:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Iris Dataset"):
                with st.spinner("Loading Iris dataset..."):
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    df = pd.DataFrame(iris.data, columns=iris.feature_names)
                    df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                    
                    # Add some missing values for demonstration
                    df.loc[df.sample(frac=0.1).index, df.columns[0]] = np.nan
                    
                    # Store data in session state
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    
                    # Store file details
                    st.session_state.file_details = {
                        'name': 'iris.csv',
                        'size': len(df) * len(df.columns) * 8,
                        'encoding': 'utf-8',
                        'confidence': 1.0,
                        'load_info': {'memory_reduction': 0.0}
                    }
                    
                    # Reset history
                    st.session_state.cleaning_history = []
                    st.session_state.command_history = []
                    st.session_state.visualization_result = None
                    
                    # Generate data insights
                    st.session_state.data_insights = st.session_state.command_processor.generate_insights(df)
                
                st.success("Loaded Iris dataset")
                st.experimental_rerun()
        
        with col2:
            if st.button("Titanic Dataset"):
                with st.spinner("Loading Titanic dataset..."):
                    # URL for the Titanic dataset
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    df = pd.read_csv(url)
                    
                    # Store data in session state
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    
                    # Store file details
                    st.session_state.file_details = {
                        'name': 'titanic.csv',
                        'size': len(df) * len(df.columns) * 8,
                        'encoding': 'utf-8',
                        'confidence': 1.0,
                        'load_info': {'memory_reduction': 0.0}
                    }
                    
                    # Reset history
                    st.session_state.cleaning_history = []
                    st.session_state.command_history = []
                    st.session_state.visualization_result = None
                    
                    # Generate data insights
                    st.session_state.data_insights = st.session_state.command_processor.generate_insights(df)
                
                st.success("Loaded Titanic dataset")
                st.experimental_rerun()
        
        with col3:
            if st.button("Boston Housing Dataset"):
                with st.spinner("Loading Boston Housing dataset..."):
                    from sklearn.datasets import load_boston
                    try:
                        boston = load_boston()
                        df = pd.DataFrame(boston.data, columns=boston.feature_names)
                        df['PRICE'] = boston.target
                    except:
                        # Alternative if load_boston is deprecated
                        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
                        df = pd.read_csv(url)
                    
                    # Add some duplicates for demonstration
                    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
                    
                    # Store data in session state
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    
                    # Store file details
                    st.session_state.file_details = {
                        'name': 'boston.csv',
                        'size': len(df) * len(df.columns) * 8,
                        'encoding': 'utf-8',
                        'confidence': 1.0,
                        'load_info': {'memory_reduction': 0.0}
                    }
                    
                    # Reset history
                    st.session_state.cleaning_history = []
                    st.session_state.command_history = []
                    st.session_state.visualization_result = None
                    
                    # Generate data insights
                    st.session_state.data_insights = st.session_state.command_processor.generate_insights(df)
                
                st.success("Loaded Boston Housing dataset")
                st.experimental_rerun()
    
    # Process command if submitted
    if 'text_command' in locals() and text_command and 'submit_button' in locals() and submit_button:
        if st.session_state.data is not None:
            with st.spinner(f"Processing command: {text_command}"):
                # Add to command history
                st.session_state.command_history.append(text_command)
                
                # Process the command
                command_result = st.session_state.command_processor.process_command(text_command)
                
                if command_result["success"]:
                    # Execute the command
                    operation = command_result['operation']
                    params = command_result['params']
                    
                    execution_result = LightweightCommandProcessor.execute_command(
                        command_result, 
                        st.session_state.data_cleaner, 
                        st.session_state.data_visualizer, 
                        st.session_state.data
                    )
                    
                    if execution_result['success']:
                        if 'data' in execution_result:
                            st.session_state.data = execution_result['data']
                        
                        if 'visualization' in execution_result:
                            st.session_state.visualization_result = execution_result['visualization']
                        
                        # Add to cleaning history
                        history_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'command': text_command,
                            'operation': operation,
                            'result': execution_result['message'],
                            'details': execution_result.get('info', {})
                        }
                        st.session_state.cleaning_history.append(history_entry)
                        
                        st.success(execution_result['message'])
                    else:
                        st.error(execution_result['message'])
                else:
                    st.error(f"Command not recognized: {text_command}")
                    if 'best_match' in command_result:
                        st.info(f"Did you mean: {command_result['best_match']}?")
        else:
            st.error("Please upload a dataset first.")

# Run the main application
if __name__ == "__main__":
    main()
