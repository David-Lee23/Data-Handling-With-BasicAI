# --- START OF FILE app.py ---

import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import re
import traceback
from typing import Dict, Any, List, Tuple
from functools import lru_cache
import time

from excel_handler import extract_data_from_excel
from data_processor import clean_dataframe, transform_dataframe
from database_connector import DatabaseConnector # Now explicitly SQLite
from nlp_engine import NaturalLanguageQueryEngine

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'data')
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)) # 16 MB default
DB_PATH = os.getenv('DATABASE_PATH', 'database.db') # Use SQLite path from env
NLP_MODEL_NAME = os.getenv('NLP_MODEL_NAME', 't5-small') # Configurable LLM
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2') # Configurable Embedding model

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Allow requests from Streamlit frontend

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        logger.info(f"Created upload folder: {UPLOAD_FOLDER}")
    except OSError as e:
        logger.critical(f"Could not create upload folder {UPLOAD_FOLDER}: {e}")
        # Application cannot proceed without upload folder
        raise SystemExit(f"Error: Could not create upload folder {UPLOAD_FOLDER}") from e

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# --- Database and NLP Engine Initialization ---
try:
    db_connector = DatabaseConnector(db_path=DB_PATH)
    # Test connection on startup
    with db_connector:
        logger.info(f"Successfully connected to SQLite database: {DB_PATH}")
except Exception as e:
    logger.critical(f"Failed to initialize database connector: {e}", exc_info=True)
    # Application cannot proceed without database
    raise SystemExit(f"Error: Database connection failed: {e}") from e

try:
    # Initialize NLP engine (consider device selection based on env var/availability if needed)
    nlp_engine = NaturalLanguageQueryEngine(db_connector, embedding_model_name=EMBEDDING_MODEL_NAME, llm_model_name=NLP_MODEL_NAME)
    # Pre-load models at startup for faster query responses.
    # This increases startup time and memory usage.
    logger.info("Loading NLP models...")
    start_time = time.time()
    nlp_engine.load_models() # Load both embedding and LLM
    logger.info(f"NLP models loaded in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    logger.error(f"Failed to load NLP models: {e}", exc_info=True)
    # Decide if the app should run without NLP or exit.
    # For this app's purpose, NLP is crucial.
    raise SystemExit(f"Error: Failed to load NLP models: {e}") from e

# --- Utility Functions ---
def allowed_file(filename: str) -> bool:
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'xlsb'}

def sanitize_identifier(name: str) -> str:
    """Sanitizes a string to be used as a table or column name."""
    # Remove invalid characters
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = '_' + name
    # Convert to lowercase (SQLite is case-insensitive for identifiers unless quoted)
    return name.lower()

def map_pandas_dtype_to_sql(dtype: Any) -> str:
    """Maps pandas dtype to SQLite data type."""
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
         # SQLite stores dates/times typically as TEXT, REAL, or INTEGER
        return 'TEXT' # Store as ISO8601 strings
    elif pd.api.types.is_bool_dtype(dtype):
        return 'INTEGER' # Store as 0 or 1
    elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
        return 'TEXT'
    else:
        logger.warning(f"Unmapped dtype '{dtype}', defaulting to TEXT.")
        return 'TEXT' # Default fallback

@lru_cache(maxsize=1) # Cache the schema, invalidate by calling cache_clear() on upload
def get_database_schema() -> Dict[str, Any]:
    """Retrieves and caches the database schema (tables and columns)."""
    logger.info("Retrieving database schema...")
    database_schema = {"tables": {}}
    try:
        with db_connector:
            # Get list of tables (excluding sqlite internal tables)
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            tables_result = db_connector.fetch_data(tables_query)
            table_names = [row['name'] for row in tables_result]

            for table_name in table_names:
                # Use PRAGMA for column info in SQLite
                column_query = f"PRAGMA table_info({table_name});"
                columns_result = db_connector.fetch_data(column_query)
                # Format: [{'cid': 0, 'name': 'col1', 'type': 'INTEGER', 'notnull': 0, 'dflt_value': None, 'pk': 1}, ...]
                columns = {col['name']: col['type'] for col in columns_result}
                # Simple description, could be enhanced with comments if stored
                description = f"Table '{table_name}' with columns: {', '.join(columns.keys())} ({', '.join(columns.values())})"
                database_schema["tables"][table_name] = {
                    "columns": list(columns.keys()),
                    "types": columns,
                    "description": description
                }
        logger.info(f"Schema retrieved for tables: {list(database_schema['tables'].keys())}")
        return database_schema
    except Exception as e:
        logger.error(f"Error fetching database schema: {e}", exc_info=True)
        # Return empty schema on error, query endpoint will handle this
        return {"tables": {}}

# --- API Endpoints ---
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads, extracts data, cleans, transforms, and stores it in the database.
    """
    start_time = time.time()
    if 'excel_file' not in request.files:
        logger.warning("Upload attempt failed: No 'excel_file' part in request.")
        return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400

    file = request.files['excel_file']
    if file.filename == '':
        logger.warning("Upload attempt failed: No file selected.")
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename) # Secure the original filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

        try:
            file.save(file_path)
            logger.info(f"File '{original_filename}' saved to '{file_path}'.")

            # Extract data from Excel file
            try:
                # excel_data is Dict[str, pd.DataFrame]
                excel_data = extract_data_from_excel(file_path)
                if not excel_data:
                    # excel_handler logs specific errors
                    return jsonify({'status': 'error', 'message': 'Could not extract data from Excel file or file was empty.'}), 400
            except Exception as e:
                logger.error(f"Error extracting data from Excel '{original_filename}': {e}", exc_info=True)
                return jsonify({'status': 'error', 'message': f'Error reading Excel file: {e}'}), 500

            tables_processed = []
            all_successful = True
            with db_connector: # Use context manager for database operations
                for sheet_name, df in excel_data.items():
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' in '{original_filename}' is empty. Skipping.")
                        continue

                    logger.info(f"Processing sheet: '{sheet_name}' from '{original_filename}'...")
                    # Clean and transform the data
                    try:
                        cleaned_df = clean_dataframe(df)
                        transformed_df = transform_dataframe(cleaned_df) # Placeholder transform
                    except Exception as e:
                        logger.error(f"Error cleaning/transforming sheet '{sheet_name}': {e}", exc_info=True)
                        all_successful = False
                        continue # Skip this sheet on error

                    # Sanitize table and column names
                    base_table_name = sanitize_identifier(f"{os.path.splitext(original_filename)[0]}_{sheet_name}")
                    # Ensure table name uniqueness if needed (e.g., append timestamp or counter)
                    # For simplicity, we'll overwrite if the same file/sheet is uploaded again.
                    table_name = base_table_name

                    # Rename columns after cleaning/transforming
                    transformed_df.columns = [sanitize_identifier(col) for col in transformed_df.columns]

                    # Map pandas dtypes to SQL data types
                    column_defs = {col: map_pandas_dtype_to_sql(dtype) for col, dtype in transformed_df.dtypes.items()}

                    try:
                        # Drop existing table before creating to ensure fresh data
                        db_connector.execute_query(f"DROP TABLE IF EXISTS {table_name};")
                        # Create table
                        db_connector.create_table(table_name, column_defs)
                        # Insert data
                        db_connector.insert_data(table_name, transformed_df)
                        # Add metadata (optional, but potentially useful)
                        # db_connector.add_metadata_columns(table_name) # Consider if needed
                        # Create indexes (optional, improves query performance on large tables)
                        # Consider indexing important columns if known or based on heuristics
                        # Example: Index first few columns or text columns
                        indexable_cols = [col for col, dtype in column_defs.items() if dtype == 'TEXT' or dtype == 'INTEGER'][:3] # Index first 3 text/int cols
                        if indexable_cols:
                             db_connector.create_index(table_name, indexable_cols)

                        tables_processed.append(table_name)
                        logger.info(f"Successfully processed and stored sheet '{sheet_name}' into table '{table_name}'.")

                    except Exception as e:
                        logger.error(f"Database error processing sheet '{sheet_name}' into table '{table_name}': {e}", exc_info=True)
                        all_successful = False
                        # Attempt to clean up partially created table? Maybe just log the error.

            # Invalidate schema cache after successful upload
            if tables_processed:
                get_database_schema.cache_clear()
                logger.info("Database schema cache cleared.")

            processing_time = time.time() - start_time
            logger.info(f"Upload processing finished in {processing_time:.2f} seconds.")

            if all_successful and tables_processed:
                 return jsonify({
                    'status': 'success',
                    'message': f'File "{original_filename}" processed successfully.',
                    'tables_created': tables_processed
                }), 200
            elif tables_processed: # Partially successful
                 return jsonify({
                    'status': 'partial',
                    'message': f'File "{original_filename}" processed with some errors. See logs for details.',
                    'tables_created': tables_processed
                }), 207 # Multi-Status
            else: # Failed completely
                 return jsonify({'status': 'error', 'message': f'Failed to process any sheets from file "{original_filename}".'}), 500

        except Exception as e:
            logger.error(f"Unhandled error during upload of '{original_filename}': {e}\n{traceback.format_exc()}")
            return jsonify({'status': 'error', 'message': f'An unexpected server error occurred: {e}'}), 500
        finally:
            # Clean up the uploaded file after processing
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: '{file_path}'")
                except OSError as e:
                    logger.warning(f"Could not remove temporary file '{file_path}': {e}")
    else:
        logger.warning(f"Upload attempt failed: Invalid file type for file '{file.filename}'.")
        return jsonify({'status': 'error', 'message': 'Invalid file type. Allowed types are xlsx, xls, xlsb.'}), 400

@app.route('/query', methods=['POST'])
def query_data():
    """
    Handles natural language queries, translates them to SQL, executes the query,
    and returns results along with a natural language summary.
    """
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'query' not in data or not data['query'].strip():
            logger.warning("Query attempt failed: No query provided.")
            return jsonify({'status': 'error', 'message': 'No query text provided'}), 400

        query = data['query'].strip()
        logger.info(f"Received query: '{query}'")

        # Get database schema (potentially cached)
        database_schema = get_database_schema()
        if not database_schema or not database_schema.get("tables"):
             logger.error("Query failed: Database schema is unavailable or empty.")
             return jsonify({'status': 'error', 'message': 'Database schema is not available. Have you uploaded a file?'}), 500

        # Map natural language query to SQL using NLP engine
        try:
            # Pass schema to the NLP engine
            sql_query = nlp_engine.map_natural_language_to_sql(query, database_schema)
            if not sql_query:
                logger.warning(f"Could not translate query '{query}' to SQL.")
                return jsonify({'status': 'error', 'message': 'Could not translate your question to an SQL query. Try rephrasing?'}), 400 # Bad Request might be better?
            logger.info(f"Generated SQL: {sql_query}")

            # Basic validation/safety check for generated SQL (prevent destructive commands)
            if not sql_query.strip().upper().startswith("SELECT"):
                 logger.error(f"Generated SQL is not a SELECT statement: {sql_query}")
                 return jsonify({'status': 'error', 'message': 'Generated query is not a valid SELECT statement.'}), 400

        except Exception as e:
            logger.error(f"Error mapping NL query to SQL: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': f'Error translating question to SQL: {e}'}), 500

        # Execute the SQL query
        query_results = []
        try:
            with db_connector:
                 # Fetch data returns list of dicts
                query_results = db_connector.fetch_data(sql_query)
            logger.info(f"SQL query executed successfully. Rows returned: {len(query_results)}")
        except Exception as e:
            logger.error(f"Error executing SQL query '{sql_query}': {e}", exc_info=True)
            # Provide the generated SQL in the error message for debugging
            return jsonify({'status': 'error', 'message': f"Error executing SQL query: {e}. Query: {sql_query}"}), 500

        # Generate a natural language response (basic for now)
        try:
            answer = nlp_engine.generate_natural_language_response(query, query_results)
        except Exception as e:
            logger.error(f"Error generating NL response: {e}", exc_info=True)
            answer = "Could not generate a natural language summary for the results." # Provide fallback

        processing_time = time.time() - start_time
        logger.info(f"Query processing finished in {processing_time:.2f} seconds.")

        return jsonify({
            'status': 'success',
            'query': query,
            'sql_query': sql_query,
            'data': query_results, # Data is already list of dicts from fetch_data
            'answer': answer,
        }), 200

    except Exception as e:
        logger.error(f"Unhandled error during query processing: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An unexpected server error occurred: {e}'}), 500

# --- Error Handlers ---
@app.errorhandler(413)
def request_entity_too_large(e):
    """Handles the 413 error (Request Entity Too Large)."""
    logger.warning(f"Upload failed: File exceeded maximum size limit ({app.config['MAX_CONTENT_LENGTH']} bytes).")
    return jsonify({
        'status': 'error',
        'message': f"File too large. Maximum allowed size is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
        }), 413

@app.errorhandler(404)
def not_found_error(error):
     logger.warning(f"Not Found error: {request.url}")
     return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
     # Log the actual error for server-side debugging
     logger.error(f"Internal Server Error: {error}", exc_info=True)
     # Return a generic message to the client
     return jsonify({'status': 'error', 'message': 'An internal server error occurred'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Use Waitress as a production-ready WSGI server
    try:
        from waitress import serve
        logger.info("Starting server with Waitress...")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        logger.warning("Waitress not found, falling back to Flask development server (not recommended for production).")
        # Fallback for environments where waitress might not be installed
        # Note: debug=True should be False in production environments
        app.run(host='0.0.0.0', port=5000, debug=False)
# --- END OF FILE app.py ---