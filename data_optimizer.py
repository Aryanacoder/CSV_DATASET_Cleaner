import pandas as pd
import numpy as np
import os
import sys
import gc
import psutil
from typing import Dict, List, Any, Tuple, Optional, Union

class DataOptimizer:
    """
    Class for optimizing data processing for large datasets.
    Provides functionality to reduce memory usage, implement chunking strategies,
    and optimize performance for large CSV files.
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage information in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / (1024 * 1024),  # Available memory in MB
            'total': psutil.virtual_memory().total / (1024 * 1024)  # Total memory in MB
        }
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types and converting
        categorical columns to category dtype.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        result = df.copy()
        memory_usage_before = result.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # Optimize integer columns
        int_columns = result.select_dtypes(include=['int']).columns
        for col in int_columns:
            col_min = result[col].min()
            col_max = result[col].max()
            
            # Convert to unsigned if possible
            if col_min >= 0:
                if col_max < 255:
                    result[col] = result[col].astype(np.uint8)
                elif col_max < 65535:
                    result[col] = result[col].astype(np.uint16)
                elif col_max < 4294967295:
                    result[col] = result[col].astype(np.uint32)
                else:
                    result[col] = result[col].astype(np.uint64)
            else:
                if col_min > -128 and col_max < 127:
                    result[col] = result[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    result[col] = result[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    result[col] = result[col].astype(np.int32)
                else:
                    result[col] = result[col].astype(np.int64)
        
        # Optimize float columns
        float_columns = result.select_dtypes(include=['float']).columns
        for col in float_columns:
            result[col] = pd.to_numeric(result[col], downcast='float')
        
        # Convert object columns to categories if appropriate
        object_columns = result.select_dtypes(include=['object']).columns
        for col in object_columns:
            num_unique = result[col].nunique()
            num_total = len(result)
            
            # If column has low cardinality (less than 50% unique values), convert to category
            if num_unique / num_total < 0.5:
                result[col] = result[col].astype('category')
        
        memory_usage_after = result.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        savings_percent = (1 - memory_usage_after / memory_usage_before) * 100
        
        print(f"Memory usage reduced from {memory_usage_before:.2f} MB to {memory_usage_after:.2f} MB ({savings_percent:.2f}% savings)")
        
        return result
    
    @staticmethod
    def process_in_chunks(file_path: str, chunk_size: int = 100000, 
                         encoding: str = 'utf-8', 
                         processing_func=None) -> pd.DataFrame:
        """
        Process a large CSV file in chunks to avoid memory issues.
        
        Args:
            file_path: Path to the CSV file
            chunk_size: Number of rows to process at a time
            encoding: File encoding
            processing_func: Function to apply to each chunk (if None, just concatenates)
            
        Returns:
            Processed DataFrame
        """
        # Initialize an empty list to store processed chunks
        processed_chunks = []
        
        # Process the file in chunks
        for i, chunk in enumerate(pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size, low_memory=True)):
            print(f"Processing chunk {i+1}...")
            
            # Apply processing function if provided
            if processing_func:
                processed_chunk = processing_func(chunk)
            else:
                processed_chunk = chunk
            
            # Append to list
            processed_chunks.append(processed_chunk)
            
            # Force garbage collection to free memory
            gc.collect()
        
        # Combine all processed chunks
        result = pd.concat(processed_chunks, ignore_index=True)
        
        return result
    
    @staticmethod
    def create_memory_efficient_copy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a memory-efficient copy of a DataFrame by optimizing dtypes.
        
        Args:
            df: DataFrame to copy
            
        Returns:
            Memory-efficient copy of the DataFrame
        """
        return DataOptimizer.optimize_dtypes(df)
    
    @staticmethod
    def sample_large_dataset(df: pd.DataFrame, n: int = 10000, 
                           random_state: int = 42) -> pd.DataFrame:
        """
        Create a random sample of a large dataset for faster exploration and testing.
        
        Args:
            df: DataFrame to sample
            n: Number of rows to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n:
            return df
        
        return df.sample(n=n, random_state=random_state)
    
    @staticmethod
    def get_column_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get memory usage information for each column in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        memory_usage_df = pd.DataFrame({
            'Column': memory_usage.index,
            'Memory (MB)': memory_usage.values / (1024 * 1024),
            'Memory (%)': 100 * memory_usage.values / memory_usage.sum()
        })
        
        # Sort by memory usage
        memory_usage_df = memory_usage_df.sort_values('Memory (MB)', ascending=False)
        
        return memory_usage_df
    
    @staticmethod
    def optimize_csv_reading(file_path: str, encoding: str = 'utf-8', 
                           usecols: Optional[List[str]] = None,
                           dtype: Optional[Dict[str, Any]] = None,
                           parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Optimize CSV reading for large files by specifying column types and selecting only needed columns.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding
            usecols: List of columns to read (if None, read all columns)
            dtype: Dictionary mapping column names to data types
            parse_dates: List of columns to parse as dates
            
        Returns:
            Optimized DataFrame
        """
        try:
            # First read a small sample to infer dtypes if not provided
            if dtype is None:
                sample = pd.read_csv(file_path, encoding=encoding, nrows=1000)
                
                # Generate dtype dictionary based on sample
                dtype = {}
                for col in sample.columns:
                    if col in (parse_dates or []):
                        continue  # Skip columns that will be parsed as dates
                    
                    if pd.api.types.is_integer_dtype(sample[col]):
                        # Use Int64 to handle potential NaN values
                        dtype[col] = 'Int64'
                    elif pd.api.types.is_float_dtype(sample[col]):
                        dtype[col] = 'float32'
                    elif sample[col].nunique() / len(sample) < 0.5:
                        # Low cardinality columns as category
                        dtype[col] = 'category'
            
            # Read the full file with optimized settings
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                usecols=usecols,
                dtype=dtype,
                parse_dates=parse_dates,
                low_memory=True
            )
            
            return df
            
        except Exception as e:
            print(f"Error optimizing CSV reading: {e}")
            
            # Fall back to standard reading
            return pd.read_csv(file_path, encoding=encoding, low_memory=True)
    
    @staticmethod
    def parallelize_dataframe(df: pd.DataFrame, func, n_cores: int = 4) -> pd.DataFrame:
        """
        Apply a function to a DataFrame in parallel using multiple cores.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each partition
            n_cores: Number of cores to use
            
        Returns:
            Processed DataFrame
        """
        from multiprocessing import Pool
        import numpy as np
        
        # Split the DataFrame into n_cores partitions
        df_split = np.array_split(df, n_cores)
        
        # Create a pool of workers
        pool = Pool(n_cores)
        
        # Apply the function to each partition in parallel
        df = pd.concat(pool.map(func, df_split))
        
        # Close the pool
        pool.close()
        pool.join()
        
        return df
    
    @staticmethod
    def optimize_for_large_dataset(app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize application configuration for handling large datasets.
        
        Args:
            app_config: Dictionary with application configuration
            
        Returns:
            Optimized configuration dictionary
        """
        # Set chunk size based on available memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Estimate rows per chunk based on available memory
        # Assume each row takes about 1KB (adjust based on your data)
        rows_per_mb = 1000  # Approximate rows per MB
        chunk_size = int(available_memory_mb * rows_per_mb * 0.2)  # Use 20% of available memory
        
        # Ensure chunk size is reasonable
        chunk_size = max(10000, min(chunk_size, 1000000))
        
        # Update configuration
        optimized_config = app_config.copy()
        optimized_config.update({
            'chunk_size': chunk_size,
            'use_chunking': True,
            'optimize_dtypes': True,
            'low_memory': True,
            'max_file_size_mb': int(available_memory_mb * 0.8),  # 80% of available memory
            'sample_for_preview': 1000,  # Number of rows to show in preview
            'enable_caching': True
        })
        
        return optimized_config
