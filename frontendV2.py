# --- START OF FILE frontend.py ---

import streamlit as st
import pandas as pd
# import json # No longer explicitly needed
import requests
import logging
from typing import Dict, Any

# Configure basic logging for the frontend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for the backend API URL
# Consider making this configurable via environment variables or a config file
# for more complex deployments.
BACKEND_URL = "http://localhost:5000"

def handle_api_response(response: requests.Response) -> Dict[str, Any]:
    """Handles common API response checking and JSON parsing."""
    if response.ok:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            st.error(f"Error decoding API response: {e}")
            logging.error(f"JSONDecodeError: {e} for response: {response.text}")
            return {"error": "Invalid response format from server."}
    else:
        try:
            error_data = response.json()
            message = error_data.get('message', 'Unknown error')
        except requests.exceptions.JSONDecodeError:
            message = response.text # Fallback to raw text if JSON parsing fails
        st.error(f"API Error (HTTP {response.status_code}): {message}")
        logging.error(f"API Error {response.status_code}: {message}")
        return {"error": message, "status_code": response.status_code}

def main():
    st.set_page_config(
        page_title="Excel Data Query Interface",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Excel Data Query Interface")
    st.markdown("Upload an Excel file and ask questions about its content in natural language.")

    # File Upload Section
    st.header("1. Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx, .xls, .xlsb)",
        type=['xlsx', 'xls', 'xlsb'],
        help="Select the Excel file containing the data you want to query."
    )

    if uploaded_file:
        st.write(f"Uploaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.2f} KB)")
        with st.spinner('Uploading and processing file... This may take a moment.'):
            try:
                files = {'excel_file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                upload_response = requests.post(f'{BACKEND_URL}/upload', files=files, timeout=120) # Increased timeout for large files

                result = handle_api_response(upload_response)

                if "error" not in result:
                    st.success(f"✅ {result.get('message', 'File processed successfully.')}")
                    tables = result.get('tables_created', [])
                    if tables:
                        st.info(f"Database tables created/updated: {', '.join(tables)}")
                    # Store uploaded file info in session state to avoid reprocessing if query changes
                    st.session_state['file_processed'] = True
                    st.session_state['processed_filename'] = uploaded_file.name
                else:
                    # Error already displayed by handle_api_response
                    st.session_state['file_processed'] = False


            except requests.exceptions.RequestException as e:
                st.error(f"Network error connecting to backend: {e}")
                logging.error(f"Network error during upload: {e}")
                st.session_state['file_processed'] = False
            except Exception as e:
                st.error(f"An unexpected error occurred during upload: {str(e)}")
                logging.error(f"Frontend upload error: {str(e)}", exc_info=True)
                st.session_state['file_processed'] = False

    # Query Section - Only show if a file has been successfully processed
    if st.session_state.get('file_processed', False):
        st.header("2. Query Data")
        query = st.text_input(
            "Enter your question (e.g., 'What are the total sales per region?', 'Show customer names and join dates')",
            key="query_input",
            placeholder=f"Ask about the data in '{st.session_state.get('processed_filename', 'your file')}'..."
            )

        if st.button("Execute Query", key="execute_button", disabled=not query):
            if query:
                with st.spinner('Generating SQL and fetching results...'):
                    try:
                        query_payload = {'query': query}
                        query_response = requests.post(
                            f'{BACKEND_URL}/query',
                            json=query_payload,
                            headers={'Content-Type': 'application/json'},
                            timeout=60
                        )

                        result = handle_api_response(query_response)

                        if "error" not in result:
                            st.subheader("Query Results")
                            tab1, tab2, tab3 = st.tabs(["📊 Natural Language Answer", "🔍 SQL Query", "📄 Raw Data"])

                            with tab1:
                                st.markdown(result.get('answer', "No natural language answer generated."))

                            with tab2:
                                st.code(result.get('sql_query', "# No SQL query generated."), language='sql')

                            with tab3:
                                raw_data = result.get('data', [])
                                if isinstance(raw_data, list) and raw_data:
                                    # Attempt to convert list of dicts to DataFrame
                                    try:
                                        df = pd.DataFrame(raw_data)
                                        st.dataframe(df)
                                    except Exception as df_e:
                                        st.warning(f"Could not display data as table: {df_e}")
                                        st.json(raw_data) # Fallback to JSON
                                elif isinstance(raw_data, (dict, str, int, float)):
                                     st.write(raw_data) # Display single values directly
                                elif not raw_data:
                                     st.info("The query returned no data.")
                                else:
                                    st.json(raw_data) # Fallback for other types

                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error connecting to backend: {e}")
                        logging.error(f"Network error during query: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during query: {str(e)}")
                        logging.error(f"Frontend query error: {str(e)}", exc_info=True)
            else:
                st.warning("Please enter a question to query the data.")
    elif 'processed_filename' in st.session_state:
         st.warning(f"File '{st.session_state['processed_filename']}' could not be processed. Please check the error message above or try uploading again.")


if __name__ == "__main__":
    # Initialize session state
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    if 'processed_filename' not in st.session_state:
        st.session_state['processed_filename'] = None

    main()
# --- END OF FILE frontend.py ---