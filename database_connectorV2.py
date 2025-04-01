# --- START OF FILE database_connector.py ---
import sqlite3
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Configure logging
logger = logging.getLogger(__name__) # Use module-specific logger

class DatabaseConnector:
    """
    Handles connections and interactions with a SQLite database.
    Uses context management for connection handling.
    """
    DEFAULT_DB_PATH = "database.db"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the database connector.

        Args:
            db_path (Optional[str]): Path to the SQLite database file.
                                     If None, uses 'DATABASE_PATH' env var or default.
        """
        _path = db_path or os.environ.get('DATABASE_PATH', self.DEFAULT_DB_PATH)
        self.db_path = Path(_path).resolve() # Use absolute path
        self.conn: Optional[sqlite3.Connection] = None
        self.is_connected: bool = False # Track connection status

        logger.info(f"DatabaseConnector initialized for SQLite database at: {self.db_path}")
        # Ensure the directory exists
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for database {self.db_path.parent}: {e}")
            # Depending on requirements, could raise an error here
            raise RuntimeError(f"Cannot create database directory: {e}") from e


    def _connect(self) -> None:
        """Establishes a connection to the SQLite database."""
        if self.is_connected and self.conn:
            # logger.debug("SQLite connection already established.")
            return
        try:
            # Consider adding timeout parameter
            self.conn = sqlite3.connect(self.db_path, timeout=10.0) # 10 second timeout
            # Set row factory to access columns by name
            self.conn.row_factory = sqlite3.Row
            # Enable foreign key support (optional, good practice if using relations)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.is_connected = True
            logger.debug(f"Established new connection to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database '{self.db_path}': {e}", exc_info=True)
            self.conn = None
            self.is_connected = False
            raise # Re-raise to signal connection failure

    def _close(self) -> None:
        """Closes the database connection."""
        if self.conn and self.is_connected:
            try:
                self.conn.close()
                logger.debug("Closed SQLite connection.")
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite connection: {e}", exc_info=True)
            finally:
                self.conn = None
                self.is_connected = False
        # else: logger.debug("No active SQLite connection to close.")


    def __enter__(self):
        """Context manager entry: Establishes connection."""
        self._connect()
        return self # Return the instance itself

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Closes connection."""
        self._close()
        # Can optionally handle exceptions here, e.g., logging exc_info
        if exc_type:
            logger.debug(f"Exiting context with exception: {exc_type.__name__}", exc_info=(exc_type, exc_val, exc_tb))
        # Return False to propagate exceptions, True to suppress (not recommended generally)
        return False

    def _get_cursor(self) -> sqlite3.Cursor:
        """Ensures connection is active and returns a cursor."""
        if not self.is_connected or not self.conn:
            # Attempt to reconnect if called outside a 'with' block? Or just raise.
            logger.error("Database operation attempted without an active connection.")
            raise sqlite3.OperationalError("No active database connection.")
        return self.conn.cursor()

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> None:
        """
        Executes a single SQL query (e.g., CREATE, DROP, INSERT, UPDATE, DELETE).

        Args:
            query (str): The SQL query string.
            params (Optional[Tuple]): Parameters to bind to the query (for safety).

        Raises:
            sqlite3.Error: If the query execution fails.
        """
        logger.debug(f"Executing query: {query}" + (f" with params: {params}" if params else ""))
        try:
            cursor = self._get_cursor()
            cursor.execute(query, params or ())
            if self.conn: self.conn.commit() # Commit changes for non-SELECT queries
            logger.debug(f"Query executed successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {query}\nParams: {params}\nError: {e}", exc_info=True)
            # Rollback changes on error? SQLite usually handles this per transaction.
            # if self.conn: self.conn.rollback() # Consider if needed based on transaction logic
            raise # Re-raise the specific SQLite error

    def fetch_data(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and fetches all results as a list of dictionaries.

        Args:
            query (str): The SQL SELECT query string.
            params (Optional[Tuple]): Parameters to bind to the query.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dict represents a row.
                                  Returns an empty list if no rows are found or on error.
        """
        logger.debug(f"Fetching data with query: {query}" + (f" with params: {params}" if params else ""))
        results: List[Dict[str, Any]] = []
        try:
            cursor = self._get_cursor()
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            # Convert sqlite3.Row objects to standard dictionaries
            results = [dict(row) for row in rows]
            logger.debug(f"Fetched {len(results)} rows.")
            return results
        except sqlite3.Error as e:
            logger.error(f"Error fetching data: {query}\nParams: {params}\nError: {e}", exc_info=True)
            return [] # Return empty list on error, as per original logic


    def create_table(self, table_name: str, column_definitions: Dict[str, str]) -> None:
        """
        Creates a new table in the SQLite database if it doesn't exist.
        Automatically adds a primary key column 'id'.

        Args:
            table_name (str): The name of the table (will be sanitized).
            column_definitions (Dict[str, str]): Dictionary of column names to SQL types
                                                 (e.g., {'name': 'TEXT', 'value': 'REAL'}).
                                                 An 'id' column will be added automatically.

        Raises:
            ValueError: If column_definitions is empty.
            sqlite3.Error: If table creation fails.
        """
        if not column_definitions:
             raise ValueError("Cannot create table with no column definitions.")

        sanitized_table_name = self._sanitize_identifier(table_name)
        logger.info(f"Attempting to create table '{sanitized_table_name}'...")

        cols_with_types = []
        # Add an auto-incrementing primary key if 'id' isn't explicitly defined
        if 'id' not in column_definitions:
             cols_with_types.append("id INTEGER PRIMARY KEY AUTOINCREMENT")

        for col_name, data_type in column_definitions.items():
             sanitized_col_name = self._sanitize_identifier(col_name)
             # Basic type validation/mapping for SQLite
             sqlite_type = self._validate_sql_type(data_type)
             cols_with_types.append(f'"{sanitized_col_name}" {sqlite_type}') # Quote identifiers

        columns_sql = ", ".join(cols_with_types)
        query = f'CREATE TABLE IF NOT EXISTS "{sanitized_table_name}" ({columns_sql});'

        self.execute_query(query)
        logger.info(f"Table '{sanitized_table_name}' created or already exists.")

    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Inserts data from a pandas DataFrame into the specified table.
        Handles pd.NA by converting it to None (SQL NULL).

        Args:
            table_name (str): The name of the table (will be sanitized).
            data (pd.DataFrame): DataFrame containing the data to insert. Columns should
                                 match the table definition (after sanitization).

        Raises:
            ValueError: If data is not a DataFrame or is empty.
            sqlite3.Error: If insertion fails.
        """
        if not isinstance(data, pd.DataFrame):
             raise ValueError("Input 'data' must be a pandas DataFrame.")
        if data.empty:
            logger.warning(f"No data provided to insert into table '{table_name}'.")
            return

        sanitized_table_name = self._sanitize_identifier(table_name)
        logger.info(f"Attempting to insert {len(data)} rows into table '{sanitized_table_name}'...")

        # Sanitize column names in the DataFrame to match table definition
        df_to_insert = data.copy()
        df_to_insert.columns = [self._sanitize_identifier(col) for col in df_to_insert.columns]
        columns = df_to_insert.columns.tolist()

        # Convert DataFrame to list of tuples, replacing pd.NA with None
        # Handle potential type issues during conversion (e.g., Timestamps)
        values_list = []
        for row_tuple in df_to_insert.itertuples(index=False, name=None):
             processed_row = []
             for value in row_tuple:
                 if pd.isna(value):
                     processed_row.append(None) # Map pd.NA/np.nan/None to SQL NULL
                 elif isinstance(value, pd.Timestamp):
                      # Convert Timestamps to ISO 8601 strings for TEXT columns
                      processed_row.append(value.isoformat())
                 # Add other type conversions if needed (e.g., timedelta to seconds/string)
                 else:
                     processed_row.append(value)
             values_list.append(tuple(processed_row))

        if not values_list:
             logger.warning(f"DataFrame was empty after processing for insertion into '{sanitized_table_name}'.")
             return

        placeholders = ", ".join(['?'] * len(columns))
        quoted_columns = ", ".join([f'"{col}"' for col in columns]) # Quote column names
        query = f'INSERT INTO "{sanitized_table_name}" ({quoted_columns}) VALUES ({placeholders});'

        # Use executemany for efficient bulk insertion
        try:
            cursor = self._get_cursor()
            cursor.executemany(query, values_list)
            if self.conn: self.conn.commit()
            logger.info(f"Successfully inserted {len(values_list)} rows into table '{sanitized_table_name}'.")
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into table '{sanitized_table_name}': {e}", exc_info=True)
            # Consider rollback if part of a larger transaction
            # if self.conn: self.conn.rollback()
            raise

    def create_index(self, table_name: str, column_names: List[str], index_name: Optional[str] = None) -> None:
        """
        Creates an index on specified columns of a table if it doesn't exist.

        Args:
            table_name (str): The name of the table (will be sanitized).
            column_names (List[str]): List of column names to index (will be sanitized).
            index_name (Optional[str]): Custom name for the index (will be sanitized).
                                        If None, a default name is generated.

        Raises:
            ValueError: If column_names is empty.
            sqlite3.Error: If index creation fails.
        """
        if not column_names:
             raise ValueError("Cannot create index with no column names.")

        sanitized_table_name = self._sanitize_identifier(table_name)
        sanitized_column_names = [self._sanitize_identifier(col) for col in column_names]
        quoted_column_names = ", ".join([f'"{col}"' for col in sanitized_column_names])

        if not index_name:
             # Generate a default index name
             index_name = f"idx_{sanitized_table_name}_{'_'.join(sanitized_column_names)}"
        sanitized_index_name = self._sanitize_identifier(index_name)

        logger.info(f"Attempting to create index '{sanitized_index_name}' on table '{sanitized_table_name}' columns: {sanitized_column_names}...")

        query = f'CREATE INDEX IF NOT EXISTS "{sanitized_index_name}" ON "{sanitized_table_name}" ({quoted_column_names});'

        self.execute_query(query)
        logger.info(f"Index '{sanitized_index_name}' created or already exists on table '{sanitized_table_name}'.")

    # --- Helper Methods ---
    def _sanitize_identifier(self, name: str) -> str:
        """Sanitizes a string for use as a SQLite identifier (table/column name)."""
        if not isinstance(name, str): name = str(name)
        # Remove characters not suitable for unquoted identifiers
        # Keep alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        # Prevent SQL keywords (basic list, might need expansion)
        sql_keywords = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'TRIGGER', 'ALTER', 'WHERE', 'FROM', 'GROUP', 'ORDER', 'BY', 'AND', 'OR', 'NOT', 'NULL', 'DEFAULT', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES'}
        if sanitized.upper() in sql_keywords:
             sanitized += '_col' # Append suffix if it's a keyword
        # SQLite identifiers are case-insensitive unless quoted. Stick to lowercase?
        # return sanitized.lower()
        return sanitized if sanitized else "unnamed_identifier"


    def _validate_sql_type(self, dtype: str) -> str:
        """Maps common type names to SQLite's affinity types."""
        # SQLite types: TEXT, NUMERIC, INTEGER, REAL, BLOB
        # https://www.sqlite.org/datatype3.html
        dtype_upper = dtype.upper()
        if 'INT' in dtype_upper: return 'INTEGER'
        if 'CHAR' in dtype_upper or 'CLOB' in dtype_upper or 'TEXT' in dtype_upper or 'STRING' in dtype_upper: return 'TEXT'
        if 'BLOB' in dtype_upper or dtype == '': return 'BLOB' # Empty type name map to BLOB
        if 'REAL' in dtype_upper or 'FLOA' in dtype_upper or 'DOUB' in dtype_upper: return 'REAL'
        # Default to NUMERIC which can store various formats
        # logger.debug(f"Mapping data type '{dtype}' to SQLite type 'NUMERIC'.")
        return 'NUMERIC'

    def add_metadata_columns(self, table_name: str) -> None:
        """
        (Optional) Adds common metadata columns to a table if they don't exist.
        Example columns: _source_file, _source_sheet, _import_time

        Args:
            table_name (str): Name of the table (will be sanitized).
        """
        sanitized_table_name = self._sanitize_identifier(table_name)
        logger.info(f"Checking/Adding metadata columns to table '{sanitized_table_name}'...")

        cols_to_add = {
            "_source_file": "TEXT",
            "_source_sheet": "TEXT",
            "_import_time": "TEXT" # Store as ISO8601 string
        }

        try:
             # Get existing columns
            cursor = self._get_cursor()
            cursor.execute(f'PRAGMA table_info("{sanitized_table_name}")')
            existing_columns = {row['name'].lower() for row in cursor.fetchall()} # Use lowercase for comparison

            for col_name, col_type in cols_to_add.items():
                if col_name.lower() not in existing_columns:
                     try:
                          add_col_query = f'ALTER TABLE "{sanitized_table_name}" ADD COLUMN "{col_name}" {col_type};'
                          self.execute_query(add_col_query)
                          logger.info(f"Added metadata column '{col_name}' to table '{sanitized_table_name}'.")
                     except sqlite3.Error as alter_err:
                          # ALTER TABLE ADD COLUMN might fail in some SQLite versions under certain conditions
                          logger.warning(f"Could not add metadata column '{col_name}' to table '{sanitized_table_name}': {alter_err}")

        except sqlite3.Error as e:
             logger.error(f"Error checking/adding metadata columns to table '{sanitized_table_name}': {e}", exc_info=True)
             # Don't raise here, as it's optional metadata


# --- END OF FILE database_connector.py ---