import pandas as pd
import numpy as np
import chardet
import os
import io
from typing import Dict, List, Any, Tuple, Optional, Union

class DataLoader:
    """
    Class for handling data loading operations for CSV files.
    Provides functionality to load, validate, and perform initial analysis on datasets.
    Optimized for large files with robust encoding detection.
    """
    
    @staticmethod
    def detect_encoding(file_content: bytes) -> Dict[str, Any]:
        """
        Detect the encoding of a file content.
        
        Args:
            file_content: Binary content of the file
            
        Returns:
            Dictionary with encoding information
        """
        # Use chardet to detect encoding
        result = chardet.detect(file_content)
        return result
    
    @staticmethod
    def load_csv(file_content: bytes, encoding: Optional[str] = None, 
                chunk_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a CSV file with proper encoding detection and handling.
        Optimized for large files with chunking support.
        
        Args:
            file_content: Binary content of the file
            encoding: Optional encoding to use (if None, will be auto-detected)
            chunk_size: Optional chunk size for reading large files
            
        Returns:
            Tuple containing:
                - DataFrame with the loaded data
                - Dictionary with metadata about the dataset
        """
        try:
            # If encoding is not provided, detect it
            if encoding is None:
                result = DataLoader.detect_encoding(file_content)
                encoding = result['encoding']
                confidence = result['confidence']
                print(f"Detected encoding: {encoding} with {confidence:.2f} confidence")
            
            # Try to read with detected encoding
            if chunk_size:
                # For large files, use chunking
                chunks = []
                for chunk in pd.read_csv(io.BytesIO(file_content), encoding=encoding, 
                                        chunksize=chunk_size, low_memory=True):
                    chunks.append(chunk)
                data = pd.concat(chunks, ignore_index=True)
            else:
                # For smaller files, read directly
                data = pd.read_csv(io.BytesIO(file_content), encoding=encoding, low_memory=True)
            
            # Generate metadata
            metadata = DataLoader.generate_metadata(data)
            
            return data, metadata
            
        except Exception as e:
            # If failed with detected encoding, try common encodings
            encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
            
            for enc in encodings_to_try:
                if enc != encoding:
                    try:
                        print(f"Trying with {enc} encoding...")
                        if chunk_size:
                            # For large files, use chunking
                            chunks = []
                            for chunk in pd.read_csv(io.BytesIO(file_content), encoding=enc, 
                                                    chunksize=chunk_size, low_memory=True):
                                chunks.append(chunk)
                            data = pd.concat(chunks, ignore_index=True)
                        else:
                            # For smaller files, read directly
                            data = pd.read_csv(io.BytesIO(file_content), encoding=enc, low_memory=True)
                        
                        print(f"Successfully loaded CSV with {enc} encoding")
                        
                        # Generate metadata
                        metadata = DataLoader.generate_metadata(data)
                        
                        return data, metadata
                    
                    except Exception as inner_e:
                        print(f"Error with {enc} encoding: {str(inner_e)}")
            
            # If all attempts fail
            raise Exception(f"Failed to load CSV with any encoding: {str(e)}")
    
    @staticmethod
    def generate_metadata(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive metadata about the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with metadata
        """
        # Basic information
        metadata = {
            'rows': data.shape[0],
            'columns': data.shape[1],
            'column_types': dict(data.dtypes.astype(str)),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024*1024),  # in MB
        }
        
        # Missing values analysis
        missing_values = data.isnull().sum().to_dict()
        missing_percentage = {col: (missing_values[col] / len(data) * 100) for col in missing_values}
        metadata['missing_values'] = missing_values
        metadata['missing_percentage'] = missing_percentage
        
        # Duplicate analysis
        metadata['duplicates'] = data.duplicated().sum()
        metadata['duplicate_percentage'] = (data.duplicated().sum() / len(data) * 100)
        
        # Column categorization
        metadata['numeric_columns'] = data.select_dtypes(include=['number']).columns.tolist()
        metadata['categorical_columns'] = data.select_dtypes(include=['object']).columns.tolist()
        metadata['datetime_columns'] = []  # Will be populated if datetime columns are detected
        metadata['unique_values'] = {col: data[col].nunique() for col in data.columns}
        
        # Try to detect datetime columns
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_datetime(data[col], errors='raise')
                    metadata['datetime_columns'].append(col)
                except:
                    pass
        
        return metadata
    
    @staticmethod
    def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with summary statistics and information
        """
        # Basic statistics for numeric columns
        numeric_stats = data.describe().to_dict() if not data.empty else {}
        
        # Categorical column statistics
        categorical_stats = {}
        for col in data.select_dtypes(include=['object']).columns:
            categorical_stats[col] = data[col].value_counts().head(10).to_dict()
        
        # Missing values analysis
        missing_values = data.isnull().sum().to_dict()
        missing_percentage = {col: (missing_values[col] / len(data) * 100) for col in missing_values}
        
        # Duplicate analysis
        duplicates = data.duplicated().sum()
        
        # Column correlation for numeric data
        correlation = {}
        if len(data.select_dtypes(include=['number']).columns) > 1:
            correlation = data.select_dtypes(include=['number']).corr().to_dict()
        
        return {
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats,
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'duplicates': duplicates,
            'correlation': correlation
        }
    
    @staticmethod
    def detect_potential_issues(data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect potential issues in the dataset that might need cleaning.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of dictionaries describing potential issues
        """
        issues = []
        
        # Check for missing values
        missing = data.isnull().sum()
        for col in missing[missing > 0].index:
            issues.append({
                'type': 'missing_values',
                'column': col,
                'count': missing[col],
                'percentage': (missing[col] / len(data) * 100),
                'suggestion': f"Consider handling missing values in column '{col}'"
            })
        
        # Check for duplicates
        if data.duplicated().sum() > 0:
            issues.append({
                'type': 'duplicates',
                'count': data.duplicated().sum(),
                'percentage': (data.duplicated().sum() / len(data) * 100),
                'suggestion': "Consider removing duplicate rows"
            })
        
        # Check for potential outliers in numeric columns
        for col in data.select_dtypes(include=['number']).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                issues.append({
                    'type': 'outliers',
                    'column': col,
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data) * 100),
                    'suggestion': f"Consider handling outliers in column '{col}'"
                })
        
        # Check for inconsistent data types
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if column might be numeric but stored as string
                numeric_count = sum(pd.to_numeric(data[col], errors='coerce').notnull())
                if numeric_count > 0 and numeric_count / len(data) > 0.5:
                    issues.append({
                        'type': 'inconsistent_type',
                        'column': col,
                        'current_type': 'object',
                        'suggested_type': 'numeric',
                        'suggestion': f"Column '{col}' might be numeric but stored as string"
                    })
                
                # Check if column might be datetime but stored as string
                datetime_count = sum(pd.to_datetime(data[col], errors='coerce').notnull())
                if datetime_count > 0 and datetime_count / len(data) > 0.5:
                    issues.append({
                        'type': 'inconsistent_type',
                        'column': col,
                        'current_type': 'object',
                        'suggested_type': 'datetime',
                        'suggestion': f"Column '{col}' might be datetime but stored as string"
                    })
        
        # Check for high cardinality categorical columns
        for col in data.select_dtypes(include=['object']).columns:
            unique_count = data[col].nunique()
            if unique_count > 100:
                issues.append({
                    'type': 'high_cardinality',
                    'column': col,
                    'unique_values': unique_count,
                    'suggestion': f"Column '{col}' has high cardinality ({unique_count} unique values)"
                })
        
        # Check for columns with low variance
        for col in data.select_dtypes(include=['number']).columns:
            if data[col].var() == 0:
                issues.append({
                    'type': 'zero_variance',
                    'column': col,
                    'suggestion': f"Column '{col}' has zero variance (constant value)"
                })
        
        return issues
    
    @staticmethod
    def load_large_csv_in_chunks(file_path: str, chunk_size: int = 100000, 
                               encoding: Optional[str] = None) -> pd.DataFrame:
        """
        Load a large CSV file in chunks to avoid memory issues.
        
        Args:
            file_path: Path to the CSV file
            chunk_size: Number of rows to read at a time
            encoding: File encoding (if None, will be auto-detected)
            
        Returns:
            DataFrame with the loaded data
        """
        # Detect encoding if not provided
        if encoding is None:
            with open(file_path, 'rb') as f:
                sample = f.read(1024 * 1024)  # Read first 1MB for detection
                result = chardet.detect(sample)
                encoding = result['encoding']
                print(f"Detected encoding: {encoding} with {result['confidence']:.2f} confidence")
        
        # Read file in chunks
        chunks = []
        for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size, low_memory=True):
            chunks.append(chunk)
        
        # Combine chunks
        data = pd.concat(chunks, ignore_index=True)
        return data
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        result = df.copy()
        
        # Downcast numeric columns
        for col in result.select_dtypes(include=['int']).columns:
            result[col] = pd.to_numeric(result[col], downcast='integer')
            
        for col in result.select_dtypes(include=['float']).columns:
            result[col] = pd.to_numeric(result[col], downcast='float')
        
        # Convert object columns to categories if appropriate
        for col in result.select_dtypes(include=['object']).columns:
            num_unique = result[col].nunique()
            num_total = len(result)
            if num_unique / num_total < 0.5:  # If less than 50% unique values
                result[col] = result[col].astype('category')
        
        return result
