# --- START OF FILE data_processor.py ---
import pandas as pd
import numpy as np
import logging
from typing import Any

# Configure logging
logger = logging.getLogger(__name__) # Use module-specific logger

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs essential cleaning on an input pandas DataFrame.

    Focuses on common issues like:
    - Handling mixed data types in columns.
    - Standardizing missing value representation (using pd.NA).
    - Removing fully duplicate rows.
    - Trimming whitespace from string columns.
    - Basic standardization (e.g., lowercase for strings).

    Args:
        df (pd.DataFrame): The raw DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame. Returns the original DataFrame if errors occur.
    """
    if not isinstance(df, pd.DataFrame):
         logger.error("Invalid input to clean_dataframe: Expected a pandas DataFrame.")
         # Or raise TypeError("Input must be a pandas DataFrame")
         return df # Return original to avoid downstream errors

    logger.info(f"Starting cleaning process for DataFrame with shape {df.shape}...")
    try:
        # --- Operate on a copy ---
        df_cleaned = df.copy()

        # --- Pre-computation ---
        original_rows = len(df_cleaned)

        # --- 1. Handle Column Names ---
        # Trim whitespace and convert to lowercase (optional, depends on desired style)
        df_cleaned.columns = df_cleaned.columns.str.strip() # .str.lower()
        logger.debug(f"Cleaned column names: {df_cleaned.columns.tolist()}")

        # --- 2. Remove Empty Rows/Columns (Optional but often useful) ---
        # Remove rows where all values are missing
        df_cleaned.dropna(axis=0, how='all', inplace=True)
        # Remove columns where all values are missing
        df_cleaned.dropna(axis=1, how='all', inplace=True)
        if df_cleaned.empty:
             logger.warning("DataFrame became empty after removing all-NA rows/columns.")
             return df_cleaned # Return empty df

        # --- 3. Data Type Inference and Handling ---
        # Attempt to infer better types, especially fixing mixed types if possible
        # df_cleaned = df_cleaned.infer_objects() # Tries to downcast object cols if possible
        # logger.debug("Attempted object type inference.")
        # More robust approach: Iterate columns and apply targeted conversions
        for col in df_cleaned.columns:
            # Convert to numeric if possible, coercing errors to NA
            # Use pd.NA for missing values as it's more consistent across types
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                logger.debug(f"Attempted numeric conversion for column '{col}'.")
            except (ValueError, TypeError):
                 # If conversion to numeric fails entirely, might be datetime or string
                 try:
                      # Attempt datetime conversion
                      df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                      # Convert NaT (Not a Time) to pd.NA for consistency
                      df_cleaned[col] = df_cleaned[col].replace({pd.NaT: pd.NA})
                      logger.debug(f"Attempted datetime conversion for column '{col}'.")
                 except (ValueError, TypeError):
                      # If it's not numeric or datetime, treat as string/object
                      if pd.api.types.is_string_dtype(df_cleaned[col]) or pd.api.types.is_object_dtype(df_cleaned[col]):
                           # Replace common placeholders with NA before string ops
                           common_na = ['na', 'n/a', 'null', 'none', '', 'nan', 'nat'] # Add others as needed
                           df_cleaned[col] = df_cleaned[col].astype(str).str.lower().replace(dict.fromkeys(common_na, pd.NA), regex=False)
                           # Trim whitespace AFTER NA replacement
                           df_cleaned[col] = df_cleaned[col].str.strip()
                           logger.debug(f"Processed string/object column '{col}' (whitespace, NA values).")
                      # If it's already boolean or some other type, leave it for now
                      else:
                           logger.debug(f"Column '{col}' seems to be of type {df_cleaned[col].dtype}, leaving as is.")

            # Final pass: Convert remaining numpy nans to pandas NA for consistency
            if df_cleaned[col].hasnans:
                 df_cleaned[col] = df_cleaned[col].replace({np.nan: pd.NA})


        # --- 4. Handle Missing Values (after type conversion) ---
        # Strategy: Keep pd.NA as the standard missing value indicator.
        # Downstream database insertion needs to handle pd.NA appropriately (usually map to NULL).
        # Imputation (like mean, median, mode, 'missing' string) is a form of transformation,
        # better handled in `transform_dataframe` if needed, as it alters the original data meaning.
        na_counts = df_cleaned.isna().sum()
        logger.info(f"Missing value counts after standardization:\n{na_counts[na_counts > 0]}")


        # --- 5. Remove Duplicate Rows ---
        df_cleaned.drop_duplicates(inplace=True)
        rows_after_duplicates = len(df_cleaned)
        duplicates_removed = (original_rows - df_cleaned.isna().all(axis=1).sum()) - rows_after_duplicates # Adjust for all-NA rows removed earlier
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows.")


        final_shape = df_cleaned.shape
        logger.info(f"DataFrame cleaning complete. Final shape: {final_shape}")
        if final_shape[0] == 0:
             logger.warning("DataFrame is empty after cleaning.")

        return df_cleaned

    except Exception as e:
        logger.error(f"Error during DataFrame cleaning: {e}", exc_info=True)
        # Return the original DataFrame to prevent breaking the pipeline
        # Consider returning None or raising error depending on desired strictness
        return df

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transformations to the cleaned DataFrame.

    This function is a placeholder for domain-specific transformations, such as:
    - Creating new features (e.g., calculating ratios, extracting date parts).
    - Applying specific imputation strategies if pd.NA is not desired.
    - Renaming columns to a standard format.
    - Filtering rows based on certain criteria.
    - Pivoting or melting the table structure.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame. Returns the input DataFrame if errors occur.
    """
    if not isinstance(df, pd.DataFrame):
         logger.error("Invalid input to transform_dataframe: Expected a pandas DataFrame.")
         return df # Return original

    logger.info(f"Starting transformation process for DataFrame with shape {df.shape}...")
    try:
        df_transformed = df.copy() # Operate on a copy

        # --- Example Transformations (Commented Out) ---

        # # Example 1: Extract year from a datetime column (if it exists)
        # date_cols = df_transformed.select_dtypes(include=['datetime64[ns]']).columns
        # for date_col in date_cols:
        #     if date_col in df_transformed:
        #         df_transformed[f'{date_col}_year'] = df_transformed[date_col].dt.year.astype(pd.Int64Dtype()) # Use nullable integer
        #         logger.info(f"Created '{date_col}_year' column.")

        # # Example 2: Fill remaining missing numeric values with the median (use after cleaning)
        # numeric_cols = df_transformed.select_dtypes(include=np.number).columns
        # for num_col in numeric_cols:
        #     if df_transformed[num_col].isna().any():
        #         median_val = df_transformed[num_col].median()
        #         df_transformed[num_col].fillna(median_val, inplace=True)
        #         logger.info(f"Imputed missing values in '{num_col}' with median ({median_val}).")

        # # Example 3: Rename columns based on a mapping
        # rename_map = {'old_col_name': 'new_col_name', 'another_old': 'super_new'}
        # # Only rename columns that actually exist in the DataFrame
        # actual_rename_map = {k: v for k, v in rename_map.items() if k in df_transformed.columns}
        # if actual_rename_map:
        #     df_transformed.rename(columns=actual_rename_map, inplace=True)
        #     logger.info(f"Renamed columns: {actual_rename_map}")

        # --- End of Example Transformations ---

        logger.info(f"DataFrame transformation complete. Final shape: {df_transformed.shape}")
        return df_transformed

    except Exception as e:
        logger.error(f"Error during DataFrame transformation: {e}", exc_info=True)
        # Return the original (cleaned) DataFrame on error
        return df
# --- END OF FILE data_processor.py ---