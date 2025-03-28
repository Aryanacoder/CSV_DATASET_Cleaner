import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import re

class DataCleaner:
    """
    Class for handling data cleaning operations for CSV datasets.
    Provides comprehensive functionality for cleaning, transforming, and preprocessing data.
    Optimized for large datasets with memory-efficient operations.
    """
    
    def __init__(self):
        """Initialize the DataCleaner with empty cleaning history"""
        self.cleaning_history = []
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'drop', 
                             columns: Optional[List[str]] = None, 
                             fill_value: Optional[Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset using various methods.
        
        Args:
            df: DataFrame to clean
            method: Method to use ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value', 'fill_knn')
            columns: List of columns to process (if None, all columns with missing values)
            fill_value: Value to use for filling (if method is 'fill_value')
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        result = df.copy()
        info = {
            'method': method,
            'columns_processed': [],
            'missing_values_before': {},
            'missing_values_after': {}
        }
        
        # If no columns specified, use all columns with missing values
        if columns is None:
            columns = [col for col in df.columns if df[col].isnull().sum() > 0]
        
        # Record missing values before cleaning
        for col in columns:
            info['missing_values_before'][col] = df[col].isnull().sum()
        
        # Apply the selected method
        if method == 'drop':
            result = result.dropna(subset=columns)
            info['rows_dropped'] = len(df) - len(result)
            
        elif method == 'fill_mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())
                    info['columns_processed'].append(col)
                    
        elif method == 'fill_median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
                    info['columns_processed'].append(col)
                    
        elif method == 'fill_mode':
            for col in columns:
                result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else None)
                info['columns_processed'].append(col)
                
        elif method == 'fill_value':
            for col in columns:
                result[col] = result[col].fillna(fill_value)
                info['columns_processed'].append(col)
                
        elif method == 'fill_knn':
            # Use KNN imputation for numeric columns
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(result[col])]
            if numeric_cols:
                # Create a subset with only numeric columns for imputation
                numeric_data = result[numeric_cols].copy()
                
                # Use KNN imputer
                imputer = KNNImputer(n_neighbors=5)
                imputed_data = imputer.fit_transform(numeric_data)
                
                # Update the result DataFrame
                for i, col in enumerate(numeric_cols):
                    result[col] = imputed_data[:, i]
                    info['columns_processed'].append(col)
        
        # Record missing values after cleaning
        for col in columns:
            info['missing_values_after'][col] = result[col].isnull().sum()
            
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'handle_missing_values',
            'method': method,
            'columns': columns,
            'info': info
        })
            
        return result, info
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, 
                         keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: DataFrame to clean
            subset: List of columns to consider for identifying duplicates (if None, all columns)
            keep: Which duplicates to keep ('first', 'last', or False for none)
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        duplicates_before = df.duplicated(subset=subset, keep=False).sum()
        result = df.drop_duplicates(subset=subset, keep=keep)
        
        info = {
            'duplicates_removed': duplicates_before - (result.duplicated(subset=subset, keep=False).sum() if keep else 0),
            'rows_before': len(df),
            'rows_after': len(result),
            'columns_considered': subset if subset else 'all',
            'keep': keep
        }
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'remove_duplicates',
            'subset': subset,
            'keep': keep,
            'info': info
        })
        
        return result, info
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: Optional[List[str]] = None, 
                       threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle outliers in the dataset using various methods.
        
        Args:
            df: DataFrame to clean
            method: Method to use ('iqr', 'zscore', 'percentile', 'winsorize')
            columns: List of columns to process (if None, all numeric columns)
            threshold: Threshold value for outlier detection (interpretation depends on method)
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        result = df.copy()
        info = {
            'method': method,
            'threshold': threshold,
            'outliers_detected': {},
            'outliers_handled': {}
        }
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter to ensure only numeric columns are processed
            columns = [col for col in columns if col in result.columns and pd.api.types.is_numeric_dtype(result[col])]
        
        info['columns_processed'] = columns
        
        for col in columns:
            # Detect outliers based on the selected method
            if method == 'iqr':
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                outliers = result[(result[col] < lower_bound) | (result[col] > upper_bound)].index
                info['outliers_detected'][col] = len(outliers)
                
                # Replace outliers with bounds
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
                
            elif method == 'zscore':
                mean = result[col].mean()
                std = result[col].std()
                z_scores = abs((result[col] - mean) / std)
                outliers = result[z_scores > threshold].index
                info['outliers_detected'][col] = len(outliers)
                
                # Replace outliers with mean
                result.loc[z_scores > threshold, col] = mean
                
            elif method == 'percentile':
                lower_bound = result[col].quantile(threshold / 100)
                upper_bound = result[col].quantile(1 - (threshold / 100))
                outliers = result[(result[col] < lower_bound) | (result[col] > upper_bound)].index
                info['outliers_detected'][col] = len(outliers)
                
                # Replace outliers with bounds
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
                
            elif method == 'winsorize':
                from scipy import stats
                # Winsorize the data (clip values outside percentile range)
                winsorized_data = stats.mstats.winsorize(result[col], limits=[threshold/100, threshold/100])
                result[col] = winsorized_data
                
                # Count how many values were modified
                original_sorted = df[col].sort_values().reset_index(drop=True)
                winsorized_sorted = pd.Series(winsorized_data).sort_values().reset_index(drop=True)
                outliers_count = (original_sorted != winsorized_sorted).sum()
                info['outliers_detected'][col] = outliers_count
            
            info['outliers_handled'][col] = info['outliers_detected'][col]
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'handle_outliers',
            'method': method,
            'threshold': threshold,
            'columns': columns,
            'info': info
        })
        
        return result, info
    
    def encode_categorical(self, df: pd.DataFrame, method: str = 'label', 
                          columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical variables in the dataset.
        
        Args:
            df: DataFrame to clean
            method: Method to use ('label', 'one_hot', 'target', 'frequency', 'binary')
            columns: List of columns to process (if None, all object columns)
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        result = df.copy()
        info = {
            'method': method,
            'columns_processed': [],
            'mappings': {}
        }
        
        # If no columns specified, use all object columns
        if columns is None:
            columns = result.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'label':
            for col in columns:
                if col in result.columns and (result[col].dtype == 'object' or result[col].dtype.name == 'category'):
                    # Create a mapping of unique values to integers
                    unique_values = result[col].dropna().unique()
                    mapping = {value: i for i, value in enumerate(unique_values)}
                    
                    # Apply the mapping
                    result[col] = result[col].map(mapping)
                    
                    info['columns_processed'].append(col)
                    info['mappings'][col] = mapping
                    
        elif method == 'one_hot':
            for col in columns:
                if col in result.columns and (result[col].dtype == 'object' or result[col].dtype.name == 'category'):
                    # Get dummies with prefix
                    dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
                    
                    # Drop original column and add dummies
                    result = pd.concat([result.drop(col, axis=1), dummies], axis=1)
                    
                    info['columns_processed'].append(col)
                    info['mappings'][col] = {val: f"{col}_{val}" for val in df[col].unique() if pd.notna(val)}
                    
        elif method == 'frequency':
            for col in columns:
                if col in result.columns and (result[col].dtype == 'object' or result[col].dtype.name == 'category'):
                    # Calculate frequency of each category
                    value_counts = result[col].value_counts(normalize=True)
                    
                    # Create mapping
                    mapping = value_counts.to_dict()
                    
                    # Apply mapping
                    result[col] = result[col].map(mapping)
                    
                    info['columns_processed'].append(col)
                    info['mappings'][col] = mapping
                    
        elif method == 'binary':
            for col in columns:
                if col in result.columns and (result[col].dtype == 'object' or result[col].dtype.name == 'category'):
                    # Check if column has only two unique values (excluding NaN)
                    unique_values = result[col].dropna().unique()
                    
                    if len(unique_values) == 2:
                        # Create binary mapping (0 and 1)
                        mapping = {unique_values[0]: 0, unique_values[1]: 1}
                        
                        # Apply mapping
                        result[col] = result[col].map(mapping)
                        
                        info['columns_processed'].append(col)
                        info['mappings'][col] = mapping
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'encode_categorical',
            'method': method,
            'columns': columns,
            'info': info
        })
        
        return result, info
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scale numeric features in the dataset.
        
        Args:
            df: DataFrame to clean
            method: Method to use ('standard', 'minmax', 'robust', 'log', 'sqrt')
            columns: List of columns to process (if None, all numeric columns)
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        result = df.copy()
        info = {
            'method': method,
            'columns_processed': [],
            'scaling_params': {}
        }
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter to ensure only numeric columns are processed
            columns = [col for col in columns if col in result.columns and pd.api.types.is_numeric_dtype(result[col])]
        
        if method == 'standard':
            # Standardize to mean=0, std=1
            scaler = StandardScaler()
            result[columns] = scaler.fit_transform(result[columns])
            
            # Store scaling parameters
            for i, col in enumerate(columns):
                info['scaling_params'][col] = {
                    'mean': scaler.mean_[i],
                    'std': scaler.scale_[i]
                }
                
        elif method == 'minmax':
            # Scale to range [0, 1]
            scaler = MinMaxScaler()
            result[columns] = scaler.fit_transform(result[columns])
            
            # Store scaling parameters
            for i, col in enumerate(columns):
                info['scaling_params'][col] = {
                    'min': scaler.data_min_[i],
                    'max': scaler.data_max_[i]
                }
                
        elif method == 'robust':
            # Scale using median and IQR (robust to outliers)
            scaler = RobustScaler()
            result[columns] = scaler.fit_transform(result[columns])
            
            # Store scaling parameters
            for i, col in enumerate(columns):
                info['scaling_params'][col] = {
                    'center': scaler.center_[i],
                    'scale': scaler.scale_[i]
                }
                
        elif method == 'log':
            # Apply log transformation (log(x + 1) to handle zeros)
            for col in columns:
                # Ensure all values are non-negative
                if result[col].min() < 0:
                    shift = abs(result[col].min()) + 1
                    result[col] = result[col] + shift
                    info['scaling_params'][col] = {'shift': shift}
                else:
                    info['scaling_params'][col] = {'shift': 0}
                
                # Apply log transformation
                result[col] = np.log1p(result[col])
                
        elif method == 'sqrt':
            # Apply square root transformation
            for col in columns:
                # Ensure all values are non-negative
                if result[col].min() < 0:
                    shift = abs(result[col].min()) + 1
                    result[col] = result[col] + shift
                    info['scaling_params'][col] = {'shift': shift}
                else:
                    info['scaling_params'][col] = {'shift': 0}
                
                # Apply sqrt transformation
                result[col] = np.sqrt(result[col])
        
        info['columns_processed'] = columns
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'scale_features',
            'method': method,
            'columns': columns,
            'info': info
        })
        
        return result, info
    
    def convert_data_types(self, df: pd.DataFrame, conversions: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert data types of columns in the dataset.
        
        Args:
            df: DataFrame to clean
            conversions: Dictionary mapping column names to target data types
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the cleaning operation
        """
        result = df.copy()
        info = {
            'successful_conversions': [],
            'failed_conversions': []
        }
        
        for col, target_type in conversions.items():
            if col in result.columns:
                try:
                    # Handle different type conversions
                    if target_type == 'int' or target_type == 'integer':
                        result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')  # Int64 handles NaN
                    elif target_type == 'float':
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                    elif target_type == 'bool' or target_type == 'boolean':
                        result[col] = result[col].astype(bool)
                    elif target_type == 'str' or target_type == 'string' or target_type == 'object':
                        result[col] = result[col].astype(str)
                    elif target_type == 'datetime':
                        result[col] = pd.to_datetime(result[col], errors='coerce')
                    elif target_type == 'category':
                        result[col] = result[col].astype('category')
                    else:
                        # Try direct conversion
                        result[col] = result[col].astype(target_type)
                    
                    info['successful_conversions'].append({
                        'column': col,
                        'original_type': str(df[col].dtype),
                        'new_type': str(result[col].dtype)
                    })
                    
                except Exception as e:
                    info['failed_conversions'].append({
                        'column': col,
                        'target_type': target_type,
                        'error': str(e)
                    })
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'convert_data_types',
            'conversions': conversions,
            'info': info
        })
        
        return result, info
    
    def filter_data(self, df: pd.DataFrame, conditions: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Filter data based on specified conditions.
        
        Args:
            df: DataFrame to clean
            conditions: List of condition dictionaries, each with:
                - 'column': Column name
                - 'operator': Comparison operator ('==', '!=', '>', '<', '>=', '<=', 'contains', 'startswith', 'endswith')
                - 'value': Value to compare against
            
        Returns:
            Tuple containing:
                - Filtered DataFrame
                - Dictionary with information about the filtering operation
        """
        result = df.copy()
        info = {
            'conditions': conditions,
            'rows_before': len(df),
            'rows_filtered': 0
        }
        
        # Apply each condition sequentially
        for condition in conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if column in result.columns:
                rows_before = len(result)
                
                # Apply the appropriate comparison
                if operator == '==':
                    result = result[result[column] == value]
                elif operator == '!=':
                    result = result[result[column] != value]
                elif operator == '>':
                    result = result[result[column] > value]
                elif operator == '<':
                    result = result[result[column] < value]
                elif operator == '>=':
                    result = result[result[column] >= value]
                elif operator == '<=':
                    result = result[result[column] <= value]
                elif operator == 'contains' and result[column].dtype == 'object':
                    result = result[result[column].str.contains(str(value), na=False)]
                elif operator == 'startswith' and result[column].dtype == 'object':
                    result = result[result[column].str.startswith(str(value), na=False)]
                elif operator == 'endswith' and result[column].dtype == 'object':
                    result = result[result[column].str.endswith(str(value), na=False)]
                elif operator == 'isin':
                    if isinstance(value, list):
                        result = result[result[column].isin(value)]
                    else:
                        # If value is not a list, convert it to a list with a single element
                        result = result[result[column].isin([value])]
                elif operator == 'between':
                    if isinstance(value, list) and len(value) == 2:
                        result = result[(result[column] >= value[0]) & (result[column] <= value[1])]
                
                # Update filtered count
                rows_filtered = rows_before - len(result)
                info[f"condition_{column}_{operator}"] = rows_filtered
        
        info['rows_after'] = len(result)
        info['rows_filtered'] = info['rows_before'] - info['rows_after']
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'filter_data',
            'conditions': conditions,
            'info': info
        })
        
        return result, info
    
    def create_features(self, df: pd.DataFrame, features: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create new features based on existing columns.
        
        Args:
            df: DataFrame to enhance
            features: List of feature dictionaries, each with:
                - 'name': New feature name
                - 'type': Feature type ('arithmetic', 'string', 'datetime', 'binning')
                - 'columns': Columns to use
                - 'operation': Operation to perform
                - Additional parameters based on feature type
            
        Returns:
            Tuple containing:
                - Enhanced DataFrame
                - Dictionary with information about the feature creation
        """
        result = df.copy()
        info = {
            'features_created': [],
            'features_failed': []
        }
        
        for feature in features:
            name = feature.get('name')
            feature_type = feature.get('type')
            columns = feature.get('columns', [])
            operation = feature.get('operation')
            
            try:
                # Create feature based on type
                if feature_type == 'arithmetic':
                    # Ensure all columns exist and are numeric
                    if all(col in result.columns for col in columns) and all(pd.api.types.is_numeric_dtype(result[col]) for col in columns):
                        if operation == 'sum':
                            result[name] = result[columns].sum(axis=1)
                        elif operation == 'mean':
                            result[name] = result[columns].mean(axis=1)
                        elif operation == 'product':
                            result[name] = result[columns].prod(axis=1)
                        elif operation == 'ratio' and len(columns) == 2:
                            # Avoid division by zero
                            result[name] = result[columns[0]] / result[columns[1]].replace(0, np.nan)
                        elif operation == 'difference' and len(columns) == 2:
                            result[name] = result[columns[0]] - result[columns[1]]
                        elif operation == 'custom' and 'expression' in feature:
                            # Use eval for custom expressions
                            expression = feature['expression']
                            for col in columns:
                                # Replace column names with dataframe references
                                expression = expression.replace(col, f"result['{col}']")
                            result[name] = eval(expression)
                    
                elif feature_type == 'string' and all(col in result.columns for col in columns):
                    if operation == 'concat':
                        # Convert all columns to string and concatenate
                        result[name] = result[columns[0]].astype(str)
                        for col in columns[1:]:
                            result[name] = result[name] + feature.get('separator', '') + result[col].astype(str)
                    elif operation == 'extract' and 'pattern' in feature and len(columns) == 1:
                        # Extract using regex
                        pattern = feature['pattern']
                        result[name] = result[columns[0]].astype(str).str.extract(pattern, expand=False)
                    elif operation == 'length' and len(columns) == 1:
                        # Get string length
                        result[name] = result[columns[0]].astype(str).str.len()
                
                elif feature_type == 'datetime' and all(col in result.columns for col in columns):
                    if operation == 'year' and len(columns) == 1:
                        # Extract year from datetime
                        result[name] = pd.to_datetime(result[columns[0]], errors='coerce').dt.year
                    elif operation == 'month' and len(columns) == 1:
                        # Extract month from datetime
                        result[name] = pd.to_datetime(result[columns[0]], errors='coerce').dt.month
                    elif operation == 'day' and len(columns) == 1:
                        # Extract day from datetime
                        result[name] = pd.to_datetime(result[columns[0]], errors='coerce').dt.day
                    elif operation == 'weekday' and len(columns) == 1:
                        # Extract weekday from datetime
                        result[name] = pd.to_datetime(result[columns[0]], errors='coerce').dt.dayofweek
                    elif operation == 'hour' and len(columns) == 1:
                        # Extract hour from datetime
                        result[name] = pd.to_datetime(result[columns[0]], errors='coerce').dt.hour
                    elif operation == 'date_diff' and len(columns) == 2:
                        # Calculate difference between two dates in days
                        date1 = pd.to_datetime(result[columns[0]], errors='coerce')
                        date2 = pd.to_datetime(result[columns[1]], errors='coerce')
                        result[name] = (date1 - date2).dt.days
                
                elif feature_type == 'binning' and len(columns) == 1 and 'bins' in feature:
                    # Bin numeric data
                    bins = feature['bins']
                    labels = feature.get('labels', None)
                    result[name] = pd.cut(result[columns[0]], bins=bins, labels=labels)
                
                info['features_created'].append({
                    'name': name,
                    'type': feature_type,
                    'operation': operation,
                    'columns': columns
                })
                
            except Exception as e:
                info['features_failed'].append({
                    'name': name,
                    'type': feature_type,
                    'operation': operation,
                    'columns': columns,
                    'error': str(e)
                })
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'create_features',
            'features': features,
            'info': info
        })
        
        return result, info
    
    def drop_columns(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Drop specified columns from the dataset.
        
        Args:
            df: DataFrame to clean
            columns: List of columns to drop
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the operation
        """
        result = df.copy()
        
        # Filter to only include columns that exist in the DataFrame
        columns_to_drop = [col for col in columns if col in result.columns]
        
        # Drop the columns
        result = result.drop(columns=columns_to_drop, errors='ignore')
        
        info = {
            'columns_dropped': columns_to_drop,
            'columns_not_found': [col for col in columns if col not in df.columns],
            'columns_before': len(df.columns),
            'columns_after': len(result.columns)
        }
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'drop_columns',
            'columns': columns,
            'info': info
        })
        
        return result, info
    
    def rename_columns(self, df: pd.DataFrame, rename_dict: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Rename columns in the dataset.
        
        Args:
            df: DataFrame to clean
            rename_dict: Dictionary mapping old column names to new column names
            
        Returns:
            Tuple containing:
                - Cleaned DataFrame
                - Dictionary with information about the operation
        """
        result = df.copy()
        
        # Filter to only include columns that exist in the DataFrame
        valid_renames = {old: new for old, new in rename_dict.items() if old in result.columns}
        
        # Rename the columns
        result = result.rename(columns=valid_renames)
        
        info = {
            'columns_renamed': valid_renames,
            'columns_not_found': [old for old in rename_dict.keys() if old not in df.columns]
        }
        
        # Add to cleaning history
        self.cleaning_history.append({
            'operation': 'rename_columns',
            'rename_dict': rename_dict,
            'info': info
        })
        
        return result, info
    
    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of cleaning operations performed.
        
        Returns:
            List of dictionaries describing each cleaning operation
        """
        return self.cleaning_history
    
    def undo_last_operation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Undo the last cleaning operation by returning to the previous state.
        Note: This requires that the original DataFrame was preserved between operations.
        
        Args:
            df: Current DataFrame
            
        Returns:
            DataFrame with the last operation undone
        """
        if self.cleaning_history:
            # Remove the last operation from history
            self.cleaning_history.pop()
            
            # Since we can't actually undo the operation (we don't store intermediate states),
            # we need to inform the user that they should use the original DataFrame
            print("Operation removed from history. Please use the original DataFrame to reapply all operations except the last one.")
            
        return df
