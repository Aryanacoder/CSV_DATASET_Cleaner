import re
import json
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_integration import HuggingFaceIntegration

class LightweightCommandProcessor:
    """
    Lightweight class for processing natural language commands for dataset cleaning operations.
    Uses simple NLP techniques instead of heavy transformer models.
    """
    
    def __init__(self):
        """Initialize the CommandProcessor"""
        self.command_history = []
        self.vectorizer = CountVectorizer(stop_words='english')
        self.command_templates = self._get_command_templates()
        self.template_texts = list(self.command_templates.keys())
        
        # Fit vectorizer on command templates
        self.vectorizer.fit(self.template_texts)
        self.template_vectors = self.vectorizer.transform(self.template_texts)
        
        # Initialize HuggingFace integration
        try:
            self.hf_integration = HuggingFaceIntegration()
            self.use_hf = True
        except Exception as e:
            print(f"Warning: Could not initialize HuggingFace integration: {e}")
            self.use_hf = False
    
    def _get_command_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined command templates and their corresponding operations.
        
        Returns:
            Dictionary mapping command templates to operations
        """
        return {
            "remove missing values": {
                "command_type": "missing_values",
                "operation": "drop_missing",
                "params": {}
            },
            "drop missing values": {
                "command_type": "missing_values",
                "operation": "drop_missing",
                "params": {}
            },
            "fill missing values with mean": {
                "command_type": "missing_values",
                "operation": "fill_mean",
                "params": {}
            },
            "replace missing values with mean": {
                "command_type": "missing_values",
                "operation": "fill_mean",
                "params": {}
            },
            "fill missing values with median": {
                "command_type": "missing_values",
                "operation": "fill_median",
                "params": {}
            },
            "replace missing values with median": {
                "command_type": "missing_values",
                "operation": "fill_median",
                "params": {}
            },
            "fill missing values with mode": {
                "command_type": "missing_values",
                "operation": "fill_mode",
                "params": {}
            },
            "replace missing values with mode": {
                "command_type": "missing_values",
                "operation": "fill_mode",
                "params": {}
            },
            "remove duplicates": {
                "command_type": "duplicates",
                "operation": "remove_duplicates",
                "params": {}
            },
            "drop duplicates": {
                "command_type": "duplicates",
                "operation": "remove_duplicates",
                "params": {}
            },
            "handle outliers": {
                "command_type": "outliers",
                "operation": "handle_outliers",
                "params": {"method": "iqr"}
            },
            "remove outliers": {
                "command_type": "outliers",
                "operation": "handle_outliers",
                "params": {"method": "iqr"}
            },
            "encode categorical variables": {
                "command_type": "encoding",
                "operation": "encode_categorical",
                "params": {"method": "label"}
            },
            "one hot encode categorical variables": {
                "command_type": "encoding",
                "operation": "encode_categorical",
                "params": {"method": "one_hot"}
            },
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
            "create histogram": {
                "command_type": "visualization",
                "operation": "histogram",
                "params": {}
            },
            "plot bar chart": {
                "command_type": "visualization",
                "operation": "bar_chart",
                "params": {}
            },
            "create bar chart": {
                "command_type": "visualization",
                "operation": "bar_chart",
                "params": {}
            },
            "plot scatter plot": {
                "command_type": "visualization",
                "operation": "scatter_plot",
                "params": {}
            },
            "create scatter plot": {
                "command_type": "visualization",
                "operation": "scatter_plot",
                "params": {}
            },
            "show missing values": {
                "command_type": "visualization",
                "operation": "missing_values_chart",
                "params": {}
            },
            "visualize missing values": {
                "command_type": "visualization",
                "operation": "missing_values_chart",
                "params": {}
            },
            "scale features": {
                "command_type": "scaling",
                "operation": "scale_features",
                "params": {"method": "standard"}
            },
            "standardize features": {
                "command_type": "scaling",
                "operation": "scale_features",
                "params": {"method": "standard"}
            },
            "normalize features": {
                "command_type": "scaling",
                "operation": "scale_features",
                "params": {"method": "minmax"}
            },
            "download cleaned dataset": {
                "command_type": "export",
                "operation": "download_csv",
                "params": {}
            },
            "save cleaned dataset": {
                "command_type": "export",
                "operation": "download_csv",
                "params": {}
            },
            "export cleaned dataset": {
                "command_type": "export",
                "operation": "download_csv",
                "params": {}
            },
            "show data summary": {
                "command_type": "info",
                "operation": "data_summary",
                "params": {}
            },
            "display data statistics": {
                "command_type": "info",
                "operation": "data_statistics",
                "params": {}
            },
            "show column types": {
                "command_type": "info",
                "operation": "column_types",
                "params": {}
            },
            # Advanced data engineering commands
            "detect outliers": {
                "command_type": "outliers",
                "operation": "detect_outliers",
                "params": {"method": "iqr"}
            },
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
        # Add to command history
        self.command_history.append(command)
        
        # Try using HuggingFace integration if available
        if self.use_hf:
            try:
                hf_result = self.hf_integration.process_command(command)
                if hf_result["success"]:
                    return hf_result
            except Exception as e:
                print(f"Error using HuggingFace integration: {e}")
        
        # Fall back to basic processing if HuggingFace integration fails or is not available
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
            
            return result
        else:
            return {
                "success": False,
                "error": "Command not recognized",
                "best_match": self.template_texts[max_similarity_idx],
                "similarity": max_similarity
            }
    
    def generate_suggestions(self, df: pd.DataFrame) -> List[str]:
        """
        Generate command suggestions based on dataset characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested commands
        """
        # Try using HuggingFace integration if available
        if self.use_hf:
            try:
                return self.hf_integration.generate_command_suggestions(df)
            except Exception as e:
                print(f"Error generating suggestions with HuggingFace integration: {e}")
        
        # Fall back to basic suggestions if HuggingFace integration fails or is not available
        suggestions = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0].index.tolist()
        
        if len(missing_columns) > 0:
            suggestions.append("Remove missing values")
            suggestions.append("Fill missing values with mean")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            suggestions.append("Remove duplicates")
        
        # Check for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            suggestions.append("Encode categorical variables")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            # Suggest visualization
            if len(numeric_cols) > 0:
                suggestions.append(f"Plot histogram of {numeric_cols[0]}")
            
            if len(numeric_cols) >= 2:
                suggestions.append("Show correlation heatmap")
                suggestions.append(f"Plot scatter plot between {numeric_cols[0]} and {numeric_cols[1]}")
        
        # Add export suggestion
        suggestions.append("Download cleaned dataset")
        
        return suggestions
    
    def generate_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with insights about the dataset
        """
        # Try using HuggingFace integration if available
        if self.use_hf:
            try:
                return self.hf_integration.generate_data_insights(df)
            except Exception as e:
                print(f"Error generating insights with HuggingFace integration: {e}")
        
        # Fall back to basic insights if HuggingFace integration fails or is not available
        insights = {
            "basic_stats": {},
            "missing_values": {},
            "duplicates": {},
            "data_types": {}
        }
        
        # Basic statistics
        insights["basic_stats"] = {
            "rows": len(df),
            "columns": len(df.columns)
        }
        
        # Missing values
        missing_counts = df.isnull().sum()
        insights["missing_values"] = {
            "total": missing_counts.sum(),
            "by_column": missing_counts.to_dict()
        }
        
        # Duplicates
        insights["duplicates"] = {
            "count": df.duplicated().sum()
        }
        
        # Data types
        insights["data_types"] = {
            "numeric": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime": [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        }
        
        return insights
    
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
        # Try using HuggingFace integration if available
        try:
            from huggingface_integration import HuggingFaceIntegration
            return HuggingFaceIntegration.execute_command(command_info, data_cleaner, data_visualizer, data)
        except Exception as e:
            print(f"Error using HuggingFace integration for command execution: {e}")
        
        # Fall back to basic execution if HuggingFace integration fails or is not available
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
                if operation == "handle_outliers":
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
            
            # Handle scaling operations
            elif command_type == "scaling":
                if operation == "scale_features":
                    result_df, info = data_cleaner.scale_features(data, **params)
                    return {
                        "success": True,
                        "message": f"Scaled {len(info['columns_processed'])} numeric columns using {info['method']} scaling",
                        "data": result_df,
                        "info": info
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
            
            # Handle export operations
            elif command_type == "export":
                if operation == "download_csv":
                    return {
                        "success": True,
                        "message": "Prepared dataset for download",
                        "data": data,
                        "download": True
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
                elif operation == "data_statistics":
                    return {
                        "success": True,
                        "message": "Generated data statistics",
                        "data": data,
                        "statistics": {
                            "shape": data.shape,
                            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                            "missing": data.isnull().sum().to_dict(),
                            "duplicates": data.duplicated().sum()
                        }
                    }
                elif operation == "column_types":
                    return {
                        "success": True,
                        "message": "Generated column type information",
                        "data": data,
                        "column_types": {col: str(dtype) for col, dtype in data.dtypes.items()}
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
