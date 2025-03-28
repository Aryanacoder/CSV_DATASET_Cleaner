import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional, Union
import io
import base64

class DataVisualizer:
    """
    Class for creating visualizations from CSV datasets.
    Provides comprehensive functionality for generating various types of plots and charts.
    Optimized for large datasets with memory-efficient operations.
    """
    
    def __init__(self):
        """Initialize the DataVisualizer"""
        self.visualization_history = []
        
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 20, 
                        title: Optional[str] = None, color: str = '#4285F4') -> Dict[str, Any]:
        """
        Create a histogram for a numeric column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            bins: Number of bins for the histogram
            title: Optional title for the plot
            color: Color for the bars
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if column not in df.columns:
            return {'error': f"Column '{column}' not found in the dataset"}
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {'error': f"Column '{column}' is not numeric"}
        
        # Create histogram using plotly
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=title if title else f"Histogram of {column}",
            color_discrete_sequence=[color]
        )
        
        # Add layout improvements
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            bargap=0.1,
            template="plotly_white"
        )
        
        # Calculate basic statistics
        stats = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max()
        }
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'histogram',
            'column': column,
            'bins': bins
        })
        
        return {
            'figure': fig,
            'type': 'histogram',
            'column': column,
            'stats': stats
        }
    
    def create_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: Optional[str] = None, 
                        title: Optional[str] = None, color: str = '#4285F4', 
                        orientation: str = 'v', top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a bar chart for categorical data.
        
        Args:
            df: DataFrame containing the data
            x_column: Column name for x-axis (categorical)
            y_column: Optional column name for y-axis (if None, counts will be used)
            title: Optional title for the plot
            color: Color for the bars
            orientation: Orientation of the bars ('v' for vertical, 'h' for horizontal)
            top_n: Optional limit to show only top N categories
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if x_column not in df.columns:
            return {'error': f"Column '{x_column}' not found in the dataset"}
        
        if y_column is not None and y_column not in df.columns:
            return {'error': f"Column '{y_column}' not found in the dataset"}
        
        # Prepare data
        if y_column is None:
            # Count occurrences of each category
            data = df[x_column].value_counts().reset_index()
            data.columns = [x_column, 'count']
            y_column = 'count'
        else:
            # Use provided y_column
            data = df[[x_column, y_column]].groupby(x_column).sum().reset_index()
        
        # Limit to top N categories if specified
        if top_n is not None and len(data) > top_n:
            data = data.sort_values(y_column, ascending=False).head(top_n)
        
        # Create bar chart using plotly
        if orientation == 'v':
            fig = px.bar(
                data, 
                x=x_column,
                y=y_column,
                title=title if title else f"Bar Chart of {x_column}",
                color_discrete_sequence=[color]
            )
        else:
            fig = px.bar(
                data, 
                y=x_column,
                x=y_column,
                title=title if title else f"Bar Chart of {x_column}",
                color_discrete_sequence=[color],
                orientation='h'
            )
        
        # Add layout improvements
        fig.update_layout(
            template="plotly_white"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'bar_chart',
            'x_column': x_column,
            'y_column': y_column,
            'orientation': orientation
        })
        
        return {
            'figure': fig,
            'type': 'bar_chart',
            'x_column': x_column,
            'y_column': y_column
        }
    
    def create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                           color_column: Optional[str] = None, size_column: Optional[str] = None,
                           title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a scatter plot for two numeric columns.
        
        Args:
            df: DataFrame containing the data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            color_column: Optional column name for point colors
            size_column: Optional column name for point sizes
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if x_column not in df.columns:
            return {'error': f"Column '{x_column}' not found in the dataset"}
        
        if y_column not in df.columns:
            return {'error': f"Column '{y_column}' not found in the dataset"}
        
        if not pd.api.types.is_numeric_dtype(df[x_column]):
            return {'error': f"Column '{x_column}' is not numeric"}
        
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return {'error': f"Column '{y_column}' is not numeric"}
        
        if color_column is not None and color_column not in df.columns:
            return {'error': f"Column '{color_column}' not found in the dataset"}
        
        if size_column is not None and size_column not in df.columns:
            return {'error': f"Column '{size_column}' not found in the dataset"}
        
        # Create scatter plot using plotly
        fig = px.scatter(
            df, 
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title if title else f"Scatter Plot of {y_column} vs {x_column}",
            opacity=0.7
        )
        
        # Add layout improvements
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white"
        )
        
        # Calculate correlation
        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'scatter_plot',
            'x_column': x_column,
            'y_column': y_column,
            'color_column': color_column,
            'size_column': size_column
        })
        
        return {
            'figure': fig,
            'type': 'scatter_plot',
            'x_column': x_column,
            'y_column': y_column,
            'correlation': correlation
        }
    
    def create_line_chart(self, df: pd.DataFrame, x_column: str, y_columns: List[str], 
                         title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a line chart for one or more series.
        
        Args:
            df: DataFrame containing the data
            x_column: Column name for x-axis
            y_columns: List of column names for y-axis (multiple lines)
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if x_column not in df.columns:
            return {'error': f"Column '{x_column}' not found in the dataset"}
        
        for col in y_columns:
            if col not in df.columns:
                return {'error': f"Column '{col}' not found in the dataset"}
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {'error': f"Column '{col}' is not numeric"}
        
        # Create line chart using plotly
        fig = go.Figure()
        
        for col in y_columns:
            fig.add_trace(go.Scatter(
                x=df[x_column],
                y=df[col],
                mode='lines',
                name=col
            ))
        
        # Add layout improvements
        fig.update_layout(
            title=title if title else f"Line Chart",
            xaxis_title=x_column,
            yaxis_title="Value",
            template="plotly_white",
            legend_title="Series"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'line_chart',
            'x_column': x_column,
            'y_columns': y_columns
        })
        
        return {
            'figure': fig,
            'type': 'line_chart',
            'x_column': x_column,
            'y_columns': y_columns
        }
    
    def create_box_plot(self, df: pd.DataFrame, numeric_column: str, 
                       group_column: Optional[str] = None, 
                       title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a box plot for a numeric column, optionally grouped by a categorical column.
        
        Args:
            df: DataFrame containing the data
            numeric_column: Column name for the numeric values
            group_column: Optional column name for grouping
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if numeric_column not in df.columns:
            return {'error': f"Column '{numeric_column}' not found in the dataset"}
        
        if not pd.api.types.is_numeric_dtype(df[numeric_column]):
            return {'error': f"Column '{numeric_column}' is not numeric"}
        
        if group_column is not None and group_column not in df.columns:
            return {'error': f"Column '{group_column}' not found in the dataset"}
        
        # Create box plot using plotly
        fig = px.box(
            df, 
            y=numeric_column,
            x=group_column,
            title=title if title else f"Box Plot of {numeric_column}" + (f" by {group_column}" if group_column else "")
        )
        
        # Add layout improvements
        fig.update_layout(
            template="plotly_white"
        )
        
        # Calculate statistics
        stats = {
            'mean': df[numeric_column].mean(),
            'median': df[numeric_column].median(),
            'q1': df[numeric_column].quantile(0.25),
            'q3': df[numeric_column].quantile(0.75),
            'iqr': df[numeric_column].quantile(0.75) - df[numeric_column].quantile(0.25),
            'min': df[numeric_column].min(),
            'max': df[numeric_column].max()
        }
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'box_plot',
            'numeric_column': numeric_column,
            'group_column': group_column
        })
        
        return {
            'figure': fig,
            'type': 'box_plot',
            'numeric_column': numeric_column,
            'group_column': group_column,
            'stats': stats
        }
    
    def create_correlation_heatmap(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                                  title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame containing the data
            columns: Optional list of columns to include (if None, all numeric columns)
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        # Select numeric columns
        numeric_data = df.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return {'error': "No numeric columns found in the dataset"}
        
        # Filter columns if specified
        if columns:
            valid_columns = [col for col in columns if col in numeric_data.columns]
            if not valid_columns:
                return {'error': "None of the specified columns are numeric or exist in the dataset"}
            numeric_data = numeric_data[valid_columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=title if title else "Correlation Heatmap",
            zmin=-1,
            zmax=1
        )
        
        # Add layout improvements
        fig.update_layout(template="plotly_white")
        
        # Find highest correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'correlation_heatmap',
            'columns': list(numeric_data.columns)
        })
        
        return {
            'figure': fig,
            'type': 'heatmap',
            'columns': list(numeric_data.columns),
            'top_correlations': corr_pairs[:5]  # Top 5 correlations
        }
    
    def create_pie_chart(self, df: pd.DataFrame, column: str, 
                        title: Optional[str] = None, 
                        top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a pie chart for a categorical column.
        
        Args:
            df: DataFrame containing the data
            column: Column name for categories
            title: Optional title for the plot
            top_n: Optional limit to show only top N categories (others grouped as 'Other')
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if column not in df.columns:
            return {'error': f"Column '{column}' not found in the dataset"}
        
        # Count occurrences of each category
        value_counts = df[column].value_counts()
        
        # Limit to top N categories if specified
        if top_n is not None and len(value_counts) > top_n:
            top_values = value_counts.head(top_n)
            other_sum = value_counts.iloc[top_n:].sum()
            
            # Create a new series with top values and 'Other'
            data = pd.Series(list(top_values) + [other_sum], 
                           index=list(top_values.index) + ['Other'])
        else:
            data = value_counts
        
        # Create pie chart using plotly
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=title if title else f"Distribution of {column}"
        )
        
        # Add layout improvements
        fig.update_layout(
            template="plotly_white"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'pie_chart',
            'column': column,
            'top_n': top_n
        })
        
        return {
            'figure': fig,
            'type': 'pie_chart',
            'column': column,
            'categories': list(data.index),
            'values': list(data.values)
        }
    
    def create_histogram_2d(self, df: pd.DataFrame, x_column: str, y_column: str, 
                           bins: int = 20, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a 2D histogram (heatmap) for two numeric columns.
        
        Args:
            df: DataFrame containing the data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            bins: Number of bins for both axes
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if x_column not in df.columns:
            return {'error': f"Column '{x_column}' not found in the dataset"}
        
        if y_column not in df.columns:
            return {'error': f"Column '{y_column}' not found in the dataset"}
        
        if not pd.api.types.is_numeric_dtype(df[x_column]):
            return {'error': f"Column '{x_column}' is not numeric"}
        
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return {'error': f"Column '{y_column}' is not numeric"}
        
        # Create 2D histogram using plotly
        fig = px.density_heatmap(
            df, 
            x=x_column,
            y=y_column,
            nbinsx=bins,
            nbinsy=bins,
            title=title if title else f"2D Histogram of {y_column} vs {x_column}",
            color_continuous_scale="Viridis"
        )
        
        # Add layout improvements
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white"
        )
        
        # Calculate correlation
        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'histogram_2d',
            'x_column': x_column,
            'y_column': y_column,
            'bins': bins
        })
        
        return {
            'figure': fig,
            'type': 'histogram_2d',
            'x_column': x_column,
            'y_column': y_column,
            'correlation': correlation
        }
    
    def create_violin_plot(self, df: pd.DataFrame, numeric_column: str, 
                          group_column: Optional[str] = None, 
                          title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a violin plot for a numeric column, optionally grouped by a categorical column.
        
        Args:
            df: DataFrame containing the data
            numeric_column: Column name for the numeric values
            group_column: Optional column name for grouping
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        if numeric_column not in df.columns:
            return {'error': f"Column '{numeric_column}' not found in the dataset"}
        
        if not pd.api.types.is_numeric_dtype(df[numeric_column]):
            return {'error': f"Column '{numeric_column}' is not numeric"}
        
        if group_column is not None and group_column not in df.columns:
            return {'error': f"Column '{group_column}' not found in the dataset"}
        
        # Create violin plot using plotly
        fig = px.violin(
            df, 
            y=numeric_column,
            x=group_column,
            box=True,  # Include box plot inside violin
            points="all",  # Show all points
            title=title if title else f"Violin Plot of {numeric_column}" + (f" by {group_column}" if group_column else "")
        )
        
        # Add layout improvements
        fig.update_layout(
            template="plotly_white"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'violin_plot',
            'numeric_column': numeric_column,
            'group_column': group_column
        })
        
        return {
            'figure': fig,
            'type': 'violin_plot',
            'numeric_column': numeric_column,
            'group_column': group_column
        }
    
    def create_missing_values_chart(self, df: pd.DataFrame, 
                                   title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a bar chart showing missing values by column.
        
        Args:
            df: DataFrame containing the data
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        # Calculate missing values and percentages
        missing = df.isnull().sum()
        missing_percent = (missing / len(df) * 100).round(2)
        
        # Create a DataFrame for plotting
        missing_data = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': missing_percent.values
        })
        
        # Sort by missing values
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        
        # Filter out columns with no missing values
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        
        if missing_data.empty:
            return {'error': "No missing values found in the dataset"}
        
        # Create bar chart using plotly
        fig = px.bar(
            missing_data, 
            x='Column',
            y='Missing Values',
            text='Percentage',
            title=title if title else "Missing Values by Column",
            color='Percentage',
            color_continuous_scale="Reds"
        )
        
        # Add percentage labels
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        
        # Add layout improvements
        fig.update_layout(
            xaxis_title="Column",
            yaxis_title="Missing Values Count",
            template="plotly_white"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'missing_values_chart'
        })
        
        return {
            'figure': fig,
            'type': 'missing_values_chart',
            'missing_data': missing_data.to_dict('records')
        }
    
    def create_distribution_grid(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                               bins: int = 20, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a grid of histograms for multiple numeric columns.
        
        Args:
            df: DataFrame containing the data
            columns: Optional list of columns to include (if None, all numeric columns)
            bins: Number of bins for histograms
            title: Optional title for the plot
            
        Returns:
            Dictionary with visualization information and figure object
        """
        # Select numeric columns
        numeric_data = df.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return {'error': "No numeric columns found in the dataset"}
        
        # Filter columns if specified
        if columns:
            valid_columns = [col for col in columns if col in numeric_data.columns]
            if not valid_columns:
                return {'error': "None of the specified columns are numeric or exist in the dataset"}
            numeric_data = numeric_data[valid_columns]
        
        # Limit to first 12 columns to avoid overcrowding
        if len(numeric_data.columns) > 12:
            numeric_data = numeric_data.iloc[:, :12]
            
        # Create a grid of histograms
        fig = make_subplots(
            rows=len(numeric_data.columns), 
            cols=1,
            subplot_titles=list(numeric_data.columns)
        )
        
        for i, col in enumerate(numeric_data.columns):
            fig.add_trace(
                go.Histogram(
                    x=numeric_data[col],
                    nbinsx=bins,
                    name=col
                ),
                row=i+1, 
                col=1
            )
        
        # Add layout improvements
        fig.update_layout(
            title=title if title else "Distribution of Numeric Variables",
            showlegend=False,
            height=300 * len(numeric_data.columns),
            template="plotly_white"
        )
        
        # Add to visualization history
        self.visualization_history.append({
            'type': 'distribution_grid',
            'columns': list(numeric_data.columns),
            'bins': bins
        })
        
        return {
            'figure': fig,
            'type': 'distribution_grid',
            'columns': list(numeric_data.columns)
        }
    
    def get_visualization_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of visualizations created.
        
        Returns:
            List of dictionaries describing each visualization
        """
        return self.visualization_history
    
    def save_figure_to_html(self, fig, filename: str) -> str:
        """
        Save a plotly figure to an HTML file.
        
        Args:
            fig: Plotly figure object
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        fig.write_html(filename)
        return filename
    
    def get_figure_as_html(self, fig) -> str:
        """
        Get a plotly figure as HTML string.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            HTML string representation of the figure
        """
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def get_figure_as_image_base64(self, fig, format: str = 'png', 
                                 width: int = 800, height: int = 600) -> str:
        """
        Get a plotly figure as base64 encoded image.
        
        Args:
            fig: Plotly figure object
            format: Image format ('png', 'jpeg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Base64 encoded image string
        """
        img_bytes = fig.to_image(format=format, width=width, height=height)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/{format};base64,{img_base64}"

# Import for make_subplots
from plotly.subplots import make_subplots
