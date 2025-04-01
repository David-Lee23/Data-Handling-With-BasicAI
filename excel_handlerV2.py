# --- START OF FILE excel_handler.py ---

from typing import Dict, Generator, List, Optional, Union, Any
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__) # Use module-specific logger

# Define supported formats and their engines
# None engine means pandas uses the default based on extension (openpyxl for .xlsx)
SUPPORTED_FORMATS: Dict[str, Dict[str, Optional[str]]] = {
    '.xlsx': {'engine': 'openpyxl'}, # Be explicit for clarity
    # '.xlsm': {'engine': 'openpyxl'}, # Typically requires openpyxl
    # '.xltx': {'engine': 'openpyxl'}, # Template files
    # '.xltm': {'engine': 'openpyxl'}, # Template files with macros
    '.xls': {'engine': 'xlrd'},     # Requires xlrd
    '.xlsb': {'engine': 'pyxlsb'}   # Requires pyxlsb
}
# Note: .xlsm, .xltx, .xltm are often readable by openpyxl but might have issues with macros/specific features.
# Keep the list focused on common data-containing formats unless broader support is strictly needed.

def _check_engine_dependency(engine_name: str) -> bool:
    """Checks if the required engine library is installed."""
    if engine_name == 'pyxlsb':
        try:
            import pyxlsb # noqa: F401
            return True
        except ImportError:
            logger.error("The 'pyxlsb' library is required to read .xlsb files. Please install it using: pip install pyxlsb")
            return False
    elif engine_name == 'xlrd':
         try:
            import xlrd # noqa: F401
            # xlrd >= 2.0 only supports .xls. Check version if strictness needed.
            return True
         except ImportError:
            logger.error("The 'xlrd' library is required to read .xls files. Please install it using: pip install xlrd")
            return False
    elif engine_name == 'openpyxl':
        try:
            import openpyxl # noqa: F401
            return True
        except ImportError:
             logger.error("The 'openpyxl' library is required to read .xlsx files. Please install it using: pip install openpyxl")
             return False
    # Add checks for other engines if needed
    return True # Assume other engines (like None) don't need specific checks here

def extract_data_from_excel(file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Extracts data from all sheets of a supported Excel file into a dictionary of pandas DataFrames.

    Args:
        file_path (Union[str, Path]): The path to the Excel file.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are sheet names and values are pandas DataFrames.
                                 Returns an empty dictionary if the file is unsupported, not found,
                                 or cannot be read.
    """
    file_path = Path(file_path)
    logger.info(f"Attempting to extract data from: {file_path}")

    if not file_path.is_file():
        logger.error(f"File not found or is not a file: {file_path}")
        return {}

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        logger.error(f"Unsupported file format: '{suffix}'. Supported formats are {', '.join(SUPPORTED_FORMATS.keys())}.")
        return {}

    engine = SUPPORTED_FORMATS[suffix]['engine']
    if engine and not _check_engine_dependency(engine):
         return {} # Dependency missing, error logged in check function

    try:
        # sheet_name=None reads all sheets into a dictionary
        # Consider adding dtype='object' to prevent pandas from guessing dtypes too early,
        # allowing more controlled type conversion later. However, this can increase memory usage.
        excel_data: Dict[str, pd.DataFrame] = pd.read_excel(
            file_path,
            sheet_name=None, # Read all sheets
            engine=engine,
            # dtype='object', # Optional: Read all as string initially
        )
        logger.info(f"Successfully extracted {len(excel_data)} sheet(s) from '{file_path}' using engine '{engine or 'default'}'.")
        return excel_data

    except FileNotFoundError:
        # This case should be caught by is_file() check, but handle defensively.
        logger.error(f"File not found during read operation: {file_path}")
        return {}
    except Exception as e:
        # Catching a broad exception here as various issues can occur during file parsing
        # depending on the file content, engine, and dependencies.
        logger.error(f"Error reading Excel file '{file_path}' with engine '{engine or 'default'}': {e}", exc_info=True)
        return {}

# --- Chunked reading function remains similar, ensure it uses the helper ---

def extract_data_from_excel_in_chunks(
    file_path: Union[str, Path],
    sheet_name: Union[str, int], # Allow sheet index as well
    chunk_size: int = 10000 # Default chunk size increased
) -> Generator[pd.DataFrame, None, None]:
    """
    Extracts data from a specific sheet of an Excel file in chunks (as DataFrames).

    Useful for very large sheets that might not fit into memory.

    Args:
        file_path (Union[str, Path]): The path to the Excel file.
        sheet_name (Union[str, int]): The name or index of the sheet to read.
        chunk_size (int): The number of rows per chunk. Must be positive.

    Yields:
        pd.DataFrame: A chunk of data from the specified sheet.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If chunk_size is not positive.
        Exception: If there's an error reading the file or sheet.
    """
    file_path = Path(file_path)
    logger.info(f"Attempting to extract data in chunks from sheet '{sheet_name}' of file: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found or is not a file: {file_path}")

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        # Raising an error might be better than returning empty generator for chunking
        raise ValueError(f"Unsupported file format: '{suffix}'. Supported formats are {', '.join(SUPPORTED_FORMATS.keys())}.")

    engine = SUPPORTED_FORMATS[suffix]['engine']
    if engine and not _check_engine_dependency(engine):
         raise ImportError(f"Missing dependency for engine '{engine}' required for file '{file_path}'")

    try:
        chunk_iterator = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine=engine,
            chunksize=chunk_size,
            # dtype='object', # Optional: Read all as string initially
        )

        processed_chunks = 0
        for chunk in chunk_iterator:
            if not chunk.empty:
                yield chunk
                processed_chunks += 1
            else:
                 logger.debug(f"Encountered an empty chunk for sheet '{sheet_name}'.") # Can happen at the end

        if processed_chunks == 0:
             logger.warning(f"No data chunks found or yielded for sheet '{sheet_name}' in '{file_path}'. The sheet might be empty.")
        else:
             logger.info(f"Successfully yielded {processed_chunks} chunk(s) for sheet '{sheet_name}'.")

    except Exception as e:
        logger.error(f"Error reading Excel file '{file_path}' (sheet: '{sheet_name}') in chunks: {e}", exc_info=True)
        # Re-raise the exception to signal failure to the caller
        raise

def get_excel_sheet_names(file_path: Union[str, Path]) -> List[str]:
    """
    Retrieves the names of all sheets in a supported Excel file.

    Args:
        file_path (Union[str, Path]): The path to the Excel file.

    Returns:
        List[str]: A list of sheet names. Returns an empty list if the file
                   is unsupported, not found, or cannot be read.
    """
    file_path = Path(file_path)
    logger.info(f"Attempting to get sheet names from: {file_path}")

    if not file_path.is_file():
        logger.error(f"File not found or is not a file: {file_path}")
        return []

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        logger.error(f"Unsupported file format: '{suffix}'. Supported formats are {', '.join(SUPPORTED_FORMATS.keys())}.")
        return []

    engine = SUPPORTED_FORMATS[suffix]['engine']
    # Note: We might not strictly need the engine dependency just to get sheet names
    # with ExcelFile, but it's safer if underlying libs behave differently.
    # Let's assume ExcelFile handles this gracefully or pd.read_excel below works.

    try:
        # Using pd.ExcelFile is generally efficient for getting just sheet names
        excel_file = pd.ExcelFile(file_path, engine=engine)
        sheet_names = excel_file.sheet_names
        logger.info(f"Successfully retrieved sheet names from '{file_path}': {sheet_names}")
        return sheet_names

    except FileNotFoundError:
        logger.error(f"File not found during sheet name retrieval: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error getting sheet names from '{file_path}': {e}", exc_info=True)
        # Fallback: Try reading just the first sheet's keys if ExcelFile fails? Maybe not reliable.
        return []

# --- END OF FILE excel_handler.py ---