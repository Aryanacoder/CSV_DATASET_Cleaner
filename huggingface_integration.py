import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class HuggingFaceIntegration:
    """
    Class for integrating Hugging Face tools and models for enhanced data engineering capabilities.
    Provides lightweight alternatives that can run without heavy dependencies.
    """
    
    def __init__(self):
        """Initialize the HuggingFaceIntegration"""
        self.vectorizer = CountVectorizer(stop_words='english')
        self.command_templates = self._get_command_templates()
        self.template_texts = list(self.command_templates.keys())
        
        # Fit vectorizer on command templates
        self.vectorizer.fit(self.template_texts)
        self.template_vectors = self.vectorizer.transform(self.template_texts)
        
        # Initialize cache for model outputs
        self.cache = {}
    
    def _get_command_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined command templates for data engineering tasks.
        
        Returns:
            Dictionary mapping command templates to operations
        """
        return {
            # Basic data cleaning commands
            "remove missing values": {
                "command_type": "missing_values",
                "operation": "drop_missing",
                "params": {}
            },
            "fill missing values with mean": {
                "command_type": "missing_values",
                "operation": "fill_mean",
                "params": {}
            },
            "fill missing values with median": {
                "command_type": "missing_values",
                "operation": "fill_median",
                "params": {}
            },
            "fill missing values with mode": {
                "command_type": "missing_values",
                "operation": "fill_mode",
                "params": {}
            },
            "remove duplicates": {
                "command_type": "duplicates",
                "operation": "remove_duplicates",
                "params": {}
            },
            
            # Advanced data engineering commands
            "detect outliers": {
                "command_type": "outliers",
                "operation": "detect_outliers",
                "params": {"method": "iqr"}
            },
            "remove outliers": {
                "command_type": "outliers",
                "operation": "remove_outliers",
                "params": {"method": "iqr"}
            },
            "encode categorical variables": {
                "command_type": "encoding",
                "operation": "encode_categorical",
                "params": {"method": "label"}
            },
            "one hot encode categorical variables": {
                "command_type": "encoding",
                "operation": "one_hot_encode",
                "params": {}
            },
            "normalize numeric features": {
                "command_type": "scaling",
                "operation": "normalize_features",
                "params": {}
            },
            "standardize numeric features": {
                "command_type": "scaling",
                "operation": "standardize_features",
                "params": {}
            },
            
            # Feature engineering commands
            "create date features": {
                "command_type": "feature_engineering",
                "operation": "create_date_features",
                "params": {}
            },
            "create text features": {
                "command_type": "feature_engineering",
                "operation": "create_text_features",
                "params": {}
            },
            "create interaction features": {
                "command_type": "feature_engineering",
                "operation": "create_interaction_features",
                "params": {}
            },
            "perform feature selection": {
                "command_type": "feature_engineering",
                "operation": "feature_selection",
                "params": {"method": "correlation"}
            },
            
            # Data transformation commands
            "apply log transformation": {
                "command_type": "transformation",
                "operation": "log_transform",
                "params": {}
            },
            "apply square root transformation": {
                "command_type": "transformation",
                "operation": "sqrt_transform",
                "params": {}
            },
            "bin numeric features": {
                "command_type": "transformation",
                "operation": "bin_features",
                "params": {"bins": 5}
            },
            
            # Data quality commands
            "check data quality": {
                "command_type": "data_quality",
                "operation": "check_quality",
                "params": {}
            },
            "detect anomalies": {
                "command_type": "data_quality",
                "operation": "detect_anomalies",
                "params": {}
            },
            "validate data types": {
                "command_type": "data_quality",
                "operation": "validate_types",
                "params": {}
            },
            
            # Visualization commands
            "show correlation heatmap": {
                "command_type": "visualization",
                "operation": "correlation_heatmap",
                "params": {}
            },
            "plot histogram": {
                "command_type": "visualization",
                "operation": "histogram",
                "params": {}
            },
            "plot bar chart": {
                "command_type": "visualization",
                "operation": "bar_chart",
                "params": {}
            },
            "plot scatter plot": {
                "command_type": "visualization",
                "operation": "scatter_plot",
                "params": {}
            },
            "show missing values": {
                "command_type": "visualization",
                "operation": "missing_values_chart",
                "params": {}
            },
            "create box plot": {
                "command_type": "visualization",
                "operation": "box_plot",
                "params": {}
            },
            "create violin plot": {
                "command_type": "visualization",
                "operation": "violin_plot",
                "params": {}
            },
            "create pair plot": {
                "command_type": "visualization",
                "operation": "pair_plot",
                "params": {}
            },
            
            # Export commands
            "download cleaned dataset": {
                "command_type": "export",
                "operation": "download_csv",
                "params": {}
            },
            "export to excel": {
                "command_type": "export",
                "operation": "export_excel",
                "params": {}
            },
            "export to json": {
                "command_type": "export",
                "operation": "export_json",
                "params": {}
            },
            
            # Information commands
            "show data summary": {
                "command_type": "info",
                "operation": "data_summary",
                "params": {}
            },
            "show column statistics": {
                "command_type": "info",
                "operation": "column_statistics",
                "params": {}
            },
            "show data types": {
                "command_type": "info",
                "operation": "data_types",
                "params": {}
            },
            "show memory usage": {
                "command_type": "info",
                "operation": "memory_usage",
                "params": {}
            }
        }
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command using lightweight NLP techniques.
        
        Args:
            command: Natural language command string
            
        Returns:
            Dictionary with processed command information
        """
        # Check cache first
        if command in self.cache:
            return self.cache[command]
        
        # Convert command to vector
        command_vector = self.vectorizer.transform([command])
        
        # Calculate cosine similarity with all templates
        similarities = cosine_similarity(command_vector, self.template_vectors)[0]
        
        # Find the most similar template
        max_similarity_idx = similarities.argmax()
        max_similarity = similarities[max_similarity_idx]
        
        # If similarity is above threshold, use the matched template
        if max_similarity > 0.5:
            matched_template = self.template_texts[max_similarity_idx]
            result = self.command_templates[matched_template].copy()
            
            # Extract column names if present in the command
            column_match = re.search(r'for (?:column|columns) ([a-zA-Z0-9_, ]+)', command, re.IGNORECASE)
            if column_match:
                columns = [col.strip() for col in column_match.group(1).split(',')]
                result["params"]["columns"] = columns
            
            # For histogram, extract column name
            if result["operation"] == "histogram" and "column" not in result["params"]:
                column_match = re.search(r'(?:of|for) ([a-zA-Z0-9_]+)', command, re.IGNORECASE)
                if column_match:
                    result["params"]["column"] = column_match.group(1).strip()
            
            # For scatter plot, extract x and y columns
            if result["operation"] == "scatter_plot":
                xy_match = re.search(r'(?:of|for|between) ([a-zA-Z0-9_]+) (?:and|vs|versus) ([a-zA-Z0-9_]+)', command, re.IGNORECASE)
                if xy_match:
                    result["params"]["x_column"] = xy_match.group(1).strip()
                    result["params"]["y_column"] = xy_match.group(2).strip()
            
            # Add success flag and similarity score
            result["success"] = True
            result["similarity"] = max_similarity
            result["matched_template"] = matched_template
            
            # Cache the result
            self.cache[command] = result
            
            return result
        else:
            result = {
                "success": False,
                "error": "Command not recognized",
                "best_match": self.template_texts[max_similarity_idx],
                "similarity": max_similarity,
                "command_type": "unknown",
                "operation": "none",
                "params": {}
            }
            
            # Cache the result
            self.cache[command] = result
            
            return result
    
    def generate_data_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate automated insights about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with insights about the dataset
        """
        insights = {
            "basic_stats": {},
            "missing_values": {},
            "correlations": {},
            "distributions": {},
            "outliers": {},
            "recommendations": []
        }
        
        # Basic statistics
        insights["basic_stats"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "duplicates": df.duplicated().sum()
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df) * 100).round(2)
        insights["missing_values"] = {
            "total_missing": missing_counts.sum(),
            "columns_with_missing": (missing_counts > 0).sum(),
            "highest_missing_column": missing_counts.idxmax() if missing_counts.max() > 0 else None,
            "highest_missing_percent": missing_percent.max()
        }
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Get top 5 correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            insights["correlations"]["top_pairs"] = corr_pairs[:5]
        
        # Distribution analysis
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            insights["distributions"][col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skew": df[col].skew(),
                "kurtosis": df[col].kurtosis()
            }
        
        # Outlier detection using IQR
        outlier_columns = {}
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_columns[col] = {
                    "count": outliers,
                    "percent": (outliers / len(df) * 100).round(2)
                }
        insights["outliers"] = outlier_columns
        
        # Generate recommendations
        if insights["missing_values"]["total_missing"] > 0:
            insights["recommendations"].append(
                f"Handle missing values in {insights['missing_values']['columns_with_missing']} columns"
            )
        
        if insights["basic_stats"]["duplicates"] > 0:
            insights["recommendations"].append(
                f"Remove {insights['basic_stats']['duplicates']} duplicate rows"
            )
        
        if outlier_columns:
            insights["recommendations"].append(
                f"Address outliers in {len(outlier_columns)} numeric columns"
            )
        
        if len(numeric_cols) > 0:
            insights["recommendations"].append(
                "Normalize or standardize numeric features for better analysis"
            )
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            insights["recommendations"].append(
                f"Encode {len(categorical_cols)} categorical columns for machine learning"
            )
        
        return insights
    
    def suggest_data_cleaning_pipeline(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Suggest an automated data cleaning pipeline based on dataset characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested cleaning steps
        """
        pipeline = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0].index.tolist()
        
        if missing_columns:
            # For columns with few missing values, suggest filling
            low_missing = [col for col in missing_columns if missing_counts[col] / len(df) < 0.1]
            if low_missing:
                numeric_low_missing = [col for col in low_missing if pd.api.types.is_numeric_dtype(df[col])]
                categorical_low_missing = [col for col in low_missing if not pd.api.types.is_numeric_dtype(df[col])]
                
                if numeric_low_missing:
                    pipeline.append({
                        "step": "fill_missing_values",
                        "method": "median",
                        "columns": numeric_low_missing,
                        "reason": "These numeric columns have few missing values (<10%)"
                    })
                
                if categorical_low_missing:
                    pipeline.append({
                        "step": "fill_missing_values",
                        "method": "mode",
                        "columns": categorical_low_missing,
                        "reason": "These categorical columns have few missing values (<10%)"
                    })
            
            # For columns with many missing values, suggest dropping
            high_missing = [col for col in missing_columns if missing_counts[col] / len(df) > 0.5]
            if high_missing:
                pipeline.append({
                    "step": "drop_columns",
                    "columns": high_missing,
                    "reason": "These columns have too many missing values (>50%)"
                })
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            pipeline.append({
                "step": "remove_duplicates",
                "reason": f"Found {duplicates} duplicate rows"
            })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        outlier_columns = []
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0 and outliers / len(df) < 0.05:  # Less than 5% are outliers
                outlier_columns.append(col)
        
        if outlier_columns:
            pipeline.append({
                "step": "handle_outliers",
                "method": "iqr",
                "columns": outlier_columns,
                "reason": "These columns contain outliers that should be addressed"
            })
        
        # Check for categorical columns that might need encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            # For low cardinality columns, suggest one-hot encoding
            low_cardinality = [col for col in categorical_cols if df[col].nunique() < 10]
            if low_cardinality:
                pipeline.append({
                    "step": "one_hot_encode",
                    "columns": low_cardinality,
                    "reason": "These categorical columns have low cardinality (<10 unique values)"
                })
            
            # For high cardinality columns, suggest label encoding
            high_cardinality = [col for col in categorical_cols if df[col].nunique() >= 10]
            if high_cardinality:
                pipeline.append({
                    "step": "label_encode",
                    "columns": high_cardinality,
                    "reason": "These categorical columns have high cardinality (≥10 unique values)"
                })
        
        # Suggest scaling numeric features
        if numeric_cols:
            # Check for skewed distributions
            skewed_cols = []
            for col in numeric_cols:
                if abs(df[col].skew()) > 1:
                    skewed_cols.append(col)
            
            if skewed_cols:
                pipeline.append({
                    "step": "transform_features",
                    "method": "log",
                    "columns": skewed_cols,
                    "reason": "These numeric columns are skewed (|skew| > 1)"
                })
            
            # Suggest standardization for remaining numeric columns
            remaining_numeric = [col for col in numeric_cols if col not in skewed_cols]
            if remaining_numeric:
                pipeline.append({
                    "step": "scale_features",
                    "method": "standard",
                    "columns": remaining_numeric,
                    "reason": "Standardize numeric features for better analysis"
                })
        
        return pipeline
    
    def generate_command_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        Generate command suggestions based on dataset characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested commands
        """
        suggestions = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0].index.tolist()
        
        if len(missing_columns) > 0:
            suggestions.append("Remove all missing values")
            suggestions.append("Fill missing values with mean")
            
            # Suggest for specific columns with high missing values
            for col in missing_columns[:2]:  # Limit to 2 columns to avoid too many suggestions
                missing_percent = missing_counts[col] / len(df) * 100
                if missing_percent > 50:
                    suggestions.append(f"Drop column {col} due to high missing values ({missing_percent:.1f}%)")
                else:
                    suggestions.append(f"Fill missing values with median for column {col}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            suggestions.append(f"Remove {duplicates} duplicate rows")
        
        # Check for categorical columns that might need encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            suggestions.append("Encode categorical variables")
            if len(categorical_cols) <= 3:
                suggestions.append(f"One hot encode categorical variables for columns {', '.join(categorical_cols)}")
        
        # Check for numeric columns that might have outliers or need scaling
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            # Suggest visualization
            if len(numeric_cols) > 0:
                suggestions.append(f"Plot histogram of {numeric_cols[0]}")
            
            if len(numeric_cols) >= 2:
                suggestions.append("Show correlation heatmap")
                suggestions.append(f"Plot scatter plot between {numeric_cols[0]} and {numeric_cols[1]}")
            
            suggestions.append("Detect outliers")
            suggestions.append("Standardize numeric features")
        
        # Check for date columns
        date_cols = []
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().any():
                    date_cols.append(col)
            except:
                pass
        
        if date_cols:
            suggestions.append(f"Create date features from {date_cols[0]}")
        
        # Check for text columns
        text_cols = []
        for col in categorical_cols:
            if df[col].astype(str).str.len().mean() > 20:  # Average length > 20 chars
                text_cols.append(col)
        
        if text_cols:
            suggestions.append(f"Create text features from {text_cols[0]}")
        
        # Add data quality suggestions
        suggestions.append("Check data quality")
        
        # Add export suggestions
        suggestions.append("Download the cleaned dataset")
        
        return suggestions
    
    @staticmethod
    def execute_command(command_info: Dict[str, Any], data_cleaner, data_visualizer, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a processed command using the appropriate data cleaning or visualization function.
        
        Args:
            command_info: Dictionary with command information
            data_cleaner: DataCleaner instance
            data_visualizer: DataVisualizer instance
            data: DataFrame to operate on
            
        Returns:
            Dictionary with execution results
        """
        if not command_info["success"]:
            return {
                "success": False,
                "message": command_info.get("error", "Command not recognized"),
                "data": data
            }
        
        command_type = command_info["command_type"]
        operation = command_info["operation"]
        params = command_info["params"]
        
        try:
            # Handle missing values operations
            if command_type == "missing_values":
                if operation == "drop_missing":
                    result_df, info = data_cleaner.handle_missing_values(data, method='drop', **params)
                    return {
                        "success": True,
                        "message": f"Dropped rows with missing values. Rows before: {len(data)}, Rows after: {len(result_df)}",
                        "data": result_df,
                        "info": info
                    }
                elif operation == "fill_mean":
                    result_df, info = data_cleaner.handle_missing_values(data, method='fill_mean', **params)
                    return {
                        "success": True,
                        "message": f"Filled missing values with mean in {len(info['columns_processed'])} columns",
                        "data": result_df,
                        "info": info
                    }
                elif operation == "fill_median":
                    result_df, info = data_cleaner.handle_missing_values(data, method='fill_median', **params)
                    return {
                        "success": True,
                        "message": f"Filled missing values with median in {len(info['columns_processed'])} columns",
                        "data": result_df,
                        "info": info
                    }
                elif operation == "fill_mode":
                    result_df, info = data_cleaner.handle_missing_values(data, method='fill_mode', **params)
                    return {
                        "success": True,
                        "message": f"Filled missing values with mode in {len(info['columns_processed'])} columns",
                        "data": result_df,
                        "info": info
                    }
            
            # Handle duplicate operations
            elif command_type == "duplicates":
                if operation == "remove_duplicates":
                    result_df, info = data_cleaner.remove_duplicates(data, **params)
                    return {
                        "success": True,
                        "message": f"Removed {info['duplicates_removed']} duplicate rows",
                        "data": result_df,
                        "info": info
                    }
            
            # Handle outlier operations
            elif command_type == "outliers":
                if operation in ["detect_outliers", "handle_outliers", "remove_outliers"]:
                    result_df, info = data_cleaner.handle_outliers(data, **params)
                    return {
                        "success": True,
                        "message": f"Handled outliers using {info['method']} method",
                        "data": result_df,
                        "info": info
                    }
            
            # Handle encoding operations
            elif command_type == "encoding":
                if operation == "encode_categorical":
                    result_df, info = data_cleaner.encode_categorical(data, **params)
                    return {
                        "success": True,
                        "message": f"Encoded {len(info['columns_processed'])} categorical columns using {info['method']} encoding",
                        "data": result_df,
                        "info": info
                    }
                elif operation == "one_hot_encode":
                    params["method"] = "one_hot"
                    result_df, info = data_cleaner.encode_categorical(data, **params)
                    return {
                        "success": True,
                        "message": f"One-hot encoded {len(info['columns_processed'])} categorical columns",
                        "data": result_df,
                        "info": info
                    }
            
            # Handle scaling operations
            elif command_type == "scaling":
                if operation == "normalize_features":
                    params["method"] = "minmax"
                    result_df, info = data_cleaner.scale_features(data, **params)
                    return {
                        "success": True,
                        "message": f"Normalized {len(info['columns_processed'])} numeric columns",
                        "data": result_df,
                        "info": info
                    }
                elif operation == "standardize_features":
                    params["method"] = "standard"
                    result_df, info = data_cleaner.scale_features(data, **params)
                    return {
                        "success": True,
                        "message": f"Standardized {len(info['columns_processed'])} numeric columns",
                        "data": result_df,
                        "info": info
                    }
            
            # Handle feature engineering operations
            elif command_type == "feature_engineering":
                if operation == "create_date_features":
                    # Find date columns
                    date_cols = []
                    for col in data.columns:
                        try:
                            if pd.to_datetime(data[col], errors='coerce').notna().any():
                                date_cols.append(col)
                        except:
                            pass
                    
                    if not date_cols:
                        return {
                            "success": False,
                            "message": "No date columns found in the dataset",
                            "data": data
                        }
                    
                    # Create date features for each date column
                    result_df = data.copy()
                    features_created = []
                    
                    for col in date_cols:
                        try:
                            # Convert to datetime
                            result_df[f"{col}_dt"] = pd.to_datetime(result_df[col], errors='coerce')
                            
                            # Extract date components
                            result_df[f"{col}_year"] = result_df[f"{col}_dt"].dt.year
                            result_df[f"{col}_month"] = result_df[f"{col}_dt"].dt.month
                            result_df[f"{col}_day"] = result_df[f"{col}_dt"].dt.day
                            result_df[f"{col}_dayofweek"] = result_df[f"{col}_dt"].dt.dayofweek
                            result_df[f"{col}_quarter"] = result_df[f"{col}_dt"].dt.quarter
                            
                            # Drop the intermediate datetime column
                            result_df = result_df.drop(f"{col}_dt", axis=1)
                            
                            features_created.extend([
                                f"{col}_year", f"{col}_month", f"{col}_day", 
                                f"{col}_dayofweek", f"{col}_quarter"
                            ])
                        except Exception as e:
                            print(f"Error creating date features for {col}: {e}")
                    
                    return {
                        "success": True,
                        "message": f"Created {len(features_created)} date features from {len(date_cols)} date columns",
                        "data": result_df,
                        "info": {
                            "date_columns": date_cols,
                            "features_created": features_created
                        }
                    }
                
                elif operation == "create_text_features":
                    # Find text columns (categorical with long values)
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    text_cols = []
                    
                    for col in categorical_cols:
                        if data[col].astype(str).str.len().mean() > 20:  # Average length > 20 chars
                            text_cols.append(col)
                    
                    if not text_cols:
                        return {
                            "success": False,
                            "message": "No text columns found in the dataset",
                            "data": data
                        }
                    
                    # Create text features for each text column
                    result_df = data.copy()
                    features_created = []
                    
                    for col in text_cols:
                        try:
                            # Character count
                            result_df[f"{col}_char_count"] = result_df[col].astype(str).str.len()
                            
                            # Word count
                            result_df[f"{col}_word_count"] = result_df[col].astype(str).str.split().str.len()
                            
                            # Uppercase count
                            result_df[f"{col}_upper_count"] = result_df[col].astype(str).str.count(r'[A-Z]')
                            
                            # Lowercase count
                            result_df[f"{col}_lower_count"] = result_df[col].astype(str).str.count(r'[a-z]')
                            
                            # Digit count
                            result_df[f"{col}_digit_count"] = result_df[col].astype(str).str.count(r'[0-9]')
                            
                            features_created.extend([
                                f"{col}_char_count", f"{col}_word_count", f"{col}_upper_count",
                                f"{col}_lower_count", f"{col}_digit_count"
                            ])
                        except Exception as e:
                            print(f"Error creating text features for {col}: {e}")
                    
                    return {
                        "success": True,
                        "message": f"Created {len(features_created)} text features from {len(text_cols)} text columns",
                        "data": result_df,
                        "info": {
                            "text_columns": text_cols,
                            "features_created": features_created
                        }
                    }
                
                elif operation == "create_interaction_features":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) < 2:
                        return {
                            "success": False,
                            "message": "Need at least 2 numeric columns to create interaction features",
                            "data": data
                        }
                    
                    # Limit to top 5 numeric columns to avoid explosion of features
                    if len(numeric_cols) > 5:
                        numeric_cols = numeric_cols[:5]
                    
                    # Create interaction features
                    result_df = data.copy()
                    features_created = []
                    
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            col1 = numeric_cols[i]
                            col2 = numeric_cols[j]
                            
                            # Multiplication
                            feature_name = f"{col1}_times_{col2}"
                            result_df[feature_name] = result_df[col1] * result_df[col2]
                            features_created.append(feature_name)
                            
                            # Division (with handling for division by zero)
                            feature_name = f"{col1}_div_{col2}"
                            result_df[feature_name] = result_df[col1] / result_df[col2].replace(0, np.nan)
                            features_created.append(feature_name)
                            
                            # Addition
                            feature_name = f"{col1}_plus_{col2}"
                            result_df[feature_name] = result_df[col1] + result_df[col2]
                            features_created.append(feature_name)
                            
                            # Subtraction
                            feature_name = f"{col1}_minus_{col2}"
                            result_df[feature_name] = result_df[col1] - result_df[col2]
                            features_created.append(feature_name)
                    
                    return {
                        "success": True,
                        "message": f"Created {len(features_created)} interaction features from {len(numeric_cols)} numeric columns",
                        "data": result_df,
                        "info": {
                            "numeric_columns": numeric_cols,
                            "features_created": features_created
                        }
                    }
                
                elif operation == "feature_selection":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) < 2:
                        return {
                            "success": False,
                            "message": "Need at least 2 numeric columns for feature selection",
                            "data": data
                        }
                    
                    method = params.get("method", "correlation")
                    
                    if method == "correlation":
                        # Calculate correlation matrix
                        corr_matrix = data[numeric_cols].corr().abs()
                        
                        # Find highly correlated features
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                        
                        if not to_drop:
                            return {
                                "success": False,
                                "message": "No highly correlated features found to drop",
                                "data": data
                            }
                        
                        # Drop highly correlated features
                        result_df = data.drop(to_drop, axis=1)
                        
                        return {
                            "success": True,
                            "message": f"Dropped {len(to_drop)} highly correlated features",
                            "data": result_df,
                            "info": {
                                "method": "correlation",
                                "dropped_features": to_drop
                            }
                        }
            
            # Handle data transformation operations
            elif command_type == "transformation":
                if operation == "log_transform":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if not numeric_cols:
                        return {
                            "success": False,
                            "message": "No numeric columns found for log transformation",
                            "data": data
                        }
                    
                    # Get columns from params or use all numeric columns
                    columns = params.get("columns", numeric_cols)
                    
                    # Apply log transformation (with handling for non-positive values)
                    result_df = data.copy()
                    transformed_cols = []
                    
                    for col in columns:
                        if col in numeric_cols:
                            # Check if column has positive values
                            if (result_df[col] <= 0).any():
                                # Add a constant to make all values positive
                                min_val = result_df[col].min()
                                shift = abs(min_val) + 1 if min_val <= 0 else 0
                                result_df[f"{col}_log"] = np.log(result_df[col] + shift)
                            else:
                                result_df[f"{col}_log"] = np.log(result_df[col])
                            
                            transformed_cols.append(col)
                    
                    return {
                        "success": True,
                        "message": f"Applied log transformation to {len(transformed_cols)} columns",
                        "data": result_df,
                        "info": {
                            "transformed_columns": transformed_cols
                        }
                    }
                
                elif operation == "sqrt_transform":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if not numeric_cols:
                        return {
                            "success": False,
                            "message": "No numeric columns found for square root transformation",
                            "data": data
                        }
                    
                    # Get columns from params or use all numeric columns
                    columns = params.get("columns", numeric_cols)
                    
                    # Apply square root transformation (with handling for negative values)
                    result_df = data.copy()
                    transformed_cols = []
                    
                    for col in columns:
                        if col in numeric_cols:
                            # Check if column has negative values
                            if (result_df[col] < 0).any():
                                # Add a constant to make all values non-negative
                                min_val = result_df[col].min()
                                shift = abs(min_val) if min_val < 0 else 0
                                result_df[f"{col}_sqrt"] = np.sqrt(result_df[col] + shift)
                            else:
                                result_df[f"{col}_sqrt"] = np.sqrt(result_df[col])
                            
                            transformed_cols.append(col)
                    
                    return {
                        "success": True,
                        "message": f"Applied square root transformation to {len(transformed_cols)} columns",
                        "data": result_df,
                        "info": {
                            "transformed_columns": transformed_cols
                        }
                    }
                
                elif operation == "bin_features":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if not numeric_cols:
                        return {
                            "success": False,
                            "message": "No numeric columns found for binning",
                            "data": data
                        }
                    
                    # Get columns from params or use all numeric columns
                    columns = params.get("columns", numeric_cols)
                    bins = params.get("bins", 5)
                    
                    # Apply binning
                    result_df = data.copy()
                    binned_cols = []
                    
                    for col in columns:
                        if col in numeric_cols:
                            result_df[f"{col}_binned"] = pd.qcut(
                                result_df[col], 
                                q=bins, 
                                labels=False, 
                                duplicates='drop'
                            )
                            binned_cols.append(col)
                    
                    return {
                        "success": True,
                        "message": f"Binned {len(binned_cols)} numeric columns into {bins} bins",
                        "data": result_df,
                        "info": {
                            "binned_columns": binned_cols,
                            "bins": bins
                        }
                    }
            
            # Handle data quality operations
            elif command_type == "data_quality":
                if operation == "check_quality":
                    # Perform comprehensive data quality check
                    quality_report = {
                        "row_count": len(data),
                        "column_count": len(data.columns),
                        "missing_values": {
                            "total": data.isnull().sum().sum(),
                            "by_column": data.isnull().sum().to_dict()
                        },
                        "duplicates": {
                            "count": data.duplicated().sum(),
                            "percentage": (data.duplicated().sum() / len(data) * 100).round(2)
                        },
                        "data_types": data.dtypes.astype(str).to_dict(),
                        "numeric_columns": {
                            "count": len(data.select_dtypes(include=['number']).columns),
                            "names": data.select_dtypes(include=['number']).columns.tolist()
                        },
                        "categorical_columns": {
                            "count": len(data.select_dtypes(include=['object', 'category']).columns),
                            "names": data.select_dtypes(include=['object', 'category']).columns.tolist()
                        },
                        "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
                    }
                    
                    # Check for outliers in numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    outlier_info = {}
                    
                    for col in numeric_cols:
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                        
                        if outliers > 0:
                            outlier_info[col] = {
                                "count": outliers,
                                "percentage": (outliers / len(data) * 100).round(2)
                            }
                    
                    quality_report["outliers"] = outlier_info
                    
                    # Check for skewed distributions
                    skew_info = {}
                    for col in numeric_cols:
                        skew_value = data[col].skew()
                        if abs(skew_value) > 1:
                            skew_info[col] = {
                                "skew": skew_value,
                                "severity": "high" if abs(skew_value) > 2 else "moderate"
                            }
                    
                    quality_report["skewed_columns"] = skew_info
                    
                    return {
                        "success": True,
                        "message": "Completed data quality check",
                        "data": data,
                        "quality_report": quality_report
                    }
                
                elif operation == "detect_anomalies":
                    # Find numeric columns
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if not numeric_cols:
                        return {
                            "success": False,
                            "message": "No numeric columns found for anomaly detection",
                            "data": data
                        }
                    
                    # Get columns from params or use all numeric columns
                    columns = params.get("columns", numeric_cols)
                    
                    # Detect anomalies using IQR method
                    result_df = data.copy()
                    anomaly_info = {}
                    
                    for col in columns:
                        if col in numeric_cols:
                            q1 = result_df[col].quantile(0.25)
                            q3 = result_df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr  # More extreme than outliers
                            upper_bound = q3 + 3 * iqr
                            
                            # Flag anomalies
                            result_df[f"{col}_anomaly"] = ((result_df[col] < lower_bound) | 
                                                         (result_df[col] > upper_bound))
                            
                            anomaly_count = result_df[f"{col}_anomaly"].sum()
                            if anomaly_count > 0:
                                anomaly_info[col] = {
                                    "count": anomaly_count,
                                    "percentage": (anomaly_count / len(result_df) * 100).round(2),
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound
                                }
                    
                    # Create overall anomaly flag
                    anomaly_cols = [col for col in result_df.columns if col.endswith('_anomaly')]
                    if anomaly_cols:
                        result_df['has_anomaly'] = result_df[anomaly_cols].any(axis=1)
                        total_anomalies = result_df['has_anomaly'].sum()
                    else:
                        total_anomalies = 0
                    
                    return {
                        "success": True,
                        "message": f"Detected {total_anomalies} rows with anomalies across {len(anomaly_info)} columns",
                        "data": result_df,
                        "info": {
                            "anomaly_details": anomaly_info,
                            "total_anomalies": total_anomalies
                        }
                    }
                
                elif operation == "validate_types":
                    # Check for potential type mismatches
                    result_df = data.copy()
                    type_issues = {}
                    
                    # Check for numeric columns stored as objects
                    for col in result_df.select_dtypes(include=['object']).columns:
                        # Try to convert to numeric
                        numeric_conversion = pd.to_numeric(result_df[col], errors='coerce')
                        if numeric_conversion.notna().sum() / len(result_df) > 0.8:  # >80% can be converted
                            type_issues[col] = {
                                "current_type": "object",
                                "suggested_type": "numeric",
                                "convertible_percentage": (numeric_conversion.notna().sum() / len(result_df) * 100).round(2)
                            }
                    
                    # Check for date columns stored as objects
                    for col in result_df.select_dtypes(include=['object']).columns:
                        if col in type_issues:
                            continue
                        
                        # Try to convert to datetime
                        date_conversion = pd.to_datetime(result_df[col], errors='coerce')
                        if date_conversion.notna().sum() / len(result_df) > 0.8:  # >80% can be converted
                            type_issues[col] = {
                                "current_type": "object",
                                "suggested_type": "datetime",
                                "convertible_percentage": (date_conversion.notna().sum() / len(result_df) * 100).round(2)
                            }
                    
                    # Check for categorical columns with low cardinality
                    for col in result_df.select_dtypes(include=['object']).columns:
                        if col in type_issues:
                            continue
                        
                        unique_ratio = result_df[col].nunique() / len(result_df)
                        if unique_ratio < 0.1:  # Less than 10% unique values
                            type_issues[col] = {
                                "current_type": "object",
                                "suggested_type": "category",
                                "unique_ratio": unique_ratio,
                                "unique_values": result_df[col].nunique()
                            }
                    
                    if not type_issues:
                        return {
                            "success": True,
                            "message": "No data type issues detected",
                            "data": result_df
                        }
                    
                    # Apply suggested type conversions
                    for col, issue in type_issues.items():
                        try:
                            if issue["suggested_type"] == "numeric":
                                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                            elif issue["suggested_type"] == "datetime":
                                result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                            elif issue["suggested_type"] == "category":
                                result_df[col] = result_df[col].astype('category')
                        except Exception as e:
                            print(f"Error converting {col}: {e}")
                    
                    return {
                        "success": True,
                        "message": f"Validated and converted data types for {len(type_issues)} columns",
                        "data": result_df,
                        "info": {
                            "type_issues": type_issues
                        }
                    }
            
            # Handle visualization operations
            elif command_type == "visualization":
                if operation == "histogram":
                    if "column" not in params:
                        return {
                            "success": False,
                            "message": "Column name is required for histogram",
                            "data": data
                        }
                    
                    viz_result = data_visualizer.create_histogram(data, **params)
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"Created histogram for column: {params['column']}",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "correlation_heatmap":
                    viz_result = data_visualizer.create_correlation_heatmap(data, **params)
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": "Created correlation heatmap",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "bar_chart":
                    if "column" not in params:
                        return {
                            "success": False,
                            "message": "Column name is required for bar chart",
                            "data": data
                        }
                    
                    viz_result = data_visualizer.create_bar_chart(data, x_column=params["column"])
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"Created bar chart for column: {params['column']}",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "scatter_plot":
                    if "x_column" not in params or "y_column" not in params:
                        return {
                            "success": False,
                            "message": "X and Y column names are required for scatter plot",
                            "data": data
                        }
                    
                    viz_result = data_visualizer.create_scatter_plot(
                        data, 
                        x_column=params["x_column"], 
                        y_column=params["y_column"]
                    )
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"Created scatter plot of {params['y_column']} vs {params['x_column']}",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "missing_values_chart":
                    viz_result = data_visualizer.create_missing_values_chart(data)
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": "Created missing values chart",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "box_plot":
                    if "column" not in params:
                        return {
                            "success": False,
                            "message": "Column name is required for box plot",
                            "data": data
                        }
                    
                    viz_result = data_visualizer.create_box_plot(data, numeric_column=params["column"])
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"Created box plot for column: {params['column']}",
                            "data": data,
                            "visualization": viz_result
                        }
                
                elif operation == "violin_plot":
                    if "column" not in params:
                        return {
                            "success": False,
                            "message": "Column name is required for violin plot",
                            "data": data
                        }
                    
                    viz_result = data_visualizer.create_violin_plot(data, numeric_column=params["column"])
                    if "error" in viz_result:
                        return {
                            "success": False,
                            "message": viz_result["error"],
                            "data": data
                        }
                    else:
                        return {
                            "success": True,
                            "message": f"Created violin plot for column: {params['column']}",
                            "data": data,
                            "visualization": viz_result
                        }
            
            # Handle export operations
            elif command_type == "export":
                if operation == "download_csv":
                    return {
                        "success": True,
                        "message": "Prepared dataset for download",
                        "data": data,
                        "download": True
                    }
                elif operation == "export_excel":
                    return {
                        "success": True,
                        "message": "Prepared dataset for Excel export",
                        "data": data,
                        "export_format": "excel"
                    }
                elif operation == "export_json":
                    return {
                        "success": True,
                        "message": "Prepared dataset for JSON export",
                        "data": data,
                        "export_format": "json"
                    }
            
            # Handle info operations
            elif command_type == "info":
                if operation == "data_summary":
                    return {
                        "success": True,
                        "message": "Generated data summary",
                        "data": data,
                        "summary": data.describe().to_dict()
                    }
                elif operation == "column_statistics":
                    return {
                        "success": True,
                        "message": "Generated column statistics",
                        "data": data,
                        "statistics": {
                            "shape": data.shape,
                            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                            "missing": data.isnull().sum().to_dict(),
                            "duplicates": data.duplicated().sum()
                        }
                    }
                elif operation == "data_types":
                    return {
                        "success": True,
                        "message": "Generated data type information",
                        "data": data,
                        "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()}
                    }
                elif operation == "memory_usage":
                    memory_usage = data.memory_usage(deep=True)
                    total_memory = memory_usage.sum() / (1024 * 1024)  # MB
                    
                    return {
                        "success": True,
                        "message": f"Total memory usage: {total_memory:.2f} MB",
                        "data": data,
                        "memory_usage": {
                            "total_mb": total_memory,
                            "by_column_mb": {col: mem / (1024 * 1024) for col, mem in memory_usage.items()}
                        }
                    }
            
            # If operation not handled
            return {
                "success": False,
                "message": f"Operation not implemented: {operation}",
                "data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "data": data
            }
