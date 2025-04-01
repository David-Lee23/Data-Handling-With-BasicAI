# --- START OF FILE nlp_engine.py ---

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Pipeline, PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import re

# Configure logging
logger = logging.getLogger(__name__) # Use module-specific logger

class NaturalLanguageQueryEngine:
    """
    Handles natural language query processing using local models:
    - Sentence Transformers for semantic search (finding relevant tables/columns).
    - Text2Text Generation models (like T5) for mapping NL query to SQL.
    """
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.65 # Adjusted threshold slightly lower based on common embedding spaces
    DEFAULT_LLM_MAX_LENGTH: int = 256 # Increased default max length for potentially complex SQL
    DEFAULT_LLM_MODEL_NAME: str = "t5-small"
    DEFAULT_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Smaller default embedding model

    def __init__(self,
                 db_connector, # Type hint omitted to avoid circular dependency if db_connector imports nlp_engine
                 device: int = -1, # -1 for CPU, >=0 for GPU ID
                 llm_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        """
        Initializes the NLP engine.

        Args:
            db_connector: Instance of DatabaseConnector (or similar interface).
            device (int): Device to run models on (-1 for CPU, >=0 for specific GPU).
                          Note: SentenceTransformer might handle device differently.
            llm_model_name (Optional[str]): Name of the Hugging Face model for text-to-SQL.
            embedding_model_name (Optional[str]): Name of the Sentence Transformer model.
        """
        self.db_connector = db_connector
        self.device = device
        self.llm_model_name = llm_model_name or self.DEFAULT_LLM_MODEL_NAME
        self.embedding_model_name = embedding_model_name or self.DEFAULT_EMBEDDING_MODEL_NAME

        self.embedding_model: Optional[SentenceTransformer] = None
        self.llm_pipeline: Optional[Pipeline] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.llm_model: Optional[PreTrainedModel] = None # Keep reference if needed

        logger.info(f"NLP Engine initialized with: LLM='{self.llm_model_name}', Embedding='{self.embedding_model_name}', Device={self.device}")

    def load_models(self) -> None:
        """Loads both the embedding and LLM models. Call this explicitly at startup if desired."""
        self._load_embedding_model()
        self._load_llm_pipeline()

    def _load_embedding_model(self) -> None:
        """Loads the Sentence Transformer model if not already loaded."""
        if self.embedding_model is None:
            try:
                logger.info(f"Loading Sentence Transformer model: '{self.embedding_model_name}'...")
                # SentenceTransformer handles device selection internally based on CUDA availability by default
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Sentence Transformer model '{self.embedding_model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Sentence Transformer model '{self.embedding_model_name}': {e}", exc_info=True)
                raise # Re-raise to signal failure

    def _load_llm_pipeline(self) -> None:
        """Loads the Text2Text Generation pipeline if not already loaded."""
        if self.llm_pipeline is None:
            try:
                logger.info(f"Loading LLM model and tokenizer: '{self.llm_model_name}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)

                logger.info(f"Creating text2text-generation pipeline for '{self.llm_model_name}' on device {self.device}...")
                # Ensure model is moved to the correct device if specified (pipeline might handle this too)
                # if self.device >= 0 and torch.cuda.is_available():
                #     self.llm_model.to(f'cuda:{self.device}')

                self.llm_pipeline = pipeline(
                    'text2text-generation',
                    model=self.llm_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    framework='pt' # Specify PyTorch framework explicitly
                )
                logger.info(f"LLM pipeline '{self.llm_model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading LLM pipeline '{self.llm_model_name}': {e}", exc_info=True)
                raise # Re-raise to signal failure

    def generate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates vector embeddings for the given text(s).
        Ensures the embedding model is loaded.

        Args:
            text (Union[str, List[str]]): A single string or a list of strings.

        Returns:
            np.ndarray: A numpy array containing the embedding(s).
        """
        self._load_embedding_model() # Ensure model is loaded
        if self.embedding_model is None:
             raise RuntimeError("Embedding model is not loaded.")

        try:
            # SentenceTransformer handles batching efficiently if text is a list
            embeddings = self.embedding_model.encode(text, show_progress_bar=False)
            # logger.debug(f"Generated embeddings for text count: {len(text) if isinstance(text, list) else 1}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise

    def map_natural_language_to_sql(self, query: str, database_schema: Dict[str, Any]) -> Optional[str]:
        """
        Maps a natural language query to an SQL query using semantic search
        to find relevant tables/columns, then an LLM to generate the SQL.

        Args:
            query (str): The natural language query.
            database_schema (dict): A dictionary representing the database schema,
                                    expected format: {"tables": {"table_name": {"columns": [], "description": "..."}}}

        Returns:
            Optional[str]: A SQL query string, or None if mapping fails.
        """
        if not query or not isinstance(query, str):
            logger.warning("Attempted to map an empty or invalid query.")
            return None
        if not database_schema or "tables" not in database_schema or not database_schema["tables"]:
             logger.warning("Attempted to map query with empty or invalid database schema.")
             return None

        logger.info(f"Mapping query to SQL: '{query}'")
        try:
            # 1. Semantic Search to find relevant tables
            relevant_tables = self._semantic_search_schema(query, database_schema)
            if not relevant_tables:
                logger.warning(f"No relevant tables found via semantic search for query: '{query}'. Attempting with all tables.")
                # Fallback: Use all tables if semantic search yields nothing (might be slow/less accurate for LLM)
                relevant_tables = list(database_schema["tables"].keys())
                if not relevant_tables:
                     logger.error("No tables available in the schema.")
                     return None # Cannot proceed without tables


            # 2. LLM-based SQL Generation using relevant context
            sql_query = self._llm_generate_sql(query, database_schema, relevant_tables)

            # 3. Basic SQL Validation (Optional but Recommended)
            validated_sql = self._validate_generated_sql(sql_query)
            if validated_sql:
                 logger.info(f"Successfully mapped query to SQL: {validated_sql}")
                 return validated_sql
            else:
                 logger.warning(f"Generated SQL failed validation: '{sql_query}'")
                 return None # Or potentially retry LLM with different prompt/params?

        except Exception as e:
            logger.error(f"Error mapping natural language query '{query}' to SQL: {e}", exc_info=True)
            return None

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors."""
        # Ensure vectors are numpy arrays
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)

        if v1.shape != v2.shape:
             logger.warning(f"Attempting cosine similarity on vectors with different shapes: {v1.shape} vs {v2.shape}")
             return 0.0 # Or raise error

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            # Handle zero vectors to avoid division by zero
            return 0.0
        else:
            similarity = dot_product / (norm_v1 * norm_v2)
            # Clip values to handle potential floating point inaccuracies
            return np.clip(similarity, -1.0, 1.0)


    def _semantic_search_schema(self, query: str, database_schema: Dict[str, Any]) -> List[str]:
        """
        Finds relevant tables (and potentially columns) based on semantic similarity
        between the query and table/column descriptions or names.

        Args:
            query (str): The natural language query.
            database_schema (Dict[str, Any]): The database schema.

        Returns:
            List[str]: A list of table names deemed most relevant to the query.
        """
        logger.debug(f"Performing semantic search for query: '{query}'")
        try:
            query_embedding = self.generate_embeddings(query)

            table_items = list(database_schema.get("tables", {}).items())
            if not table_items:
                logger.warning("No tables found in schema for semantic search.")
                return []

            # Prepare text corpus for embedding: Use table descriptions or names+columns
            corpus = []
            table_names_in_corpus = []
            for table_name, table_info in table_items:
                 # Prefer description if available and meaningful
                 description = table_info.get("description")
                 if description and len(description.split()) > 5: # Heuristic: Use description if it's somewhat detailed
                     corpus.append(description)
                 else:
                     # Fallback: Combine table name and column names
                     col_str = ", ".join(table_info.get("columns", []))
                     corpus.append(f"Table {table_name} columns: {col_str}")
                 table_names_in_corpus.append(table_name)


            if not corpus:
                 logger.warning("Could not build corpus for semantic search from schema.")
                 return []

            # Generate embeddings for the schema corpus (batched)
            corpus_embeddings = self.generate_embeddings(corpus)

            # Calculate similarities
            similarities = [self._cosine_similarity(query_embedding, doc_embedding)
                            for doc_embedding in corpus_embeddings]

            # Select tables above the threshold, sorted by similarity
            relevant_indices = np.argsort(similarities)[::-1] # Sort descending
            relevant_tables = []
            for i in relevant_indices:
                 sim = similarities[i]
                 if sim >= self.DEFAULT_SIMILARITY_THRESHOLD:
                     table_name = table_names_in_corpus[i]
                     relevant_tables.append(table_name)
                     logger.debug(f"Found relevant table '{table_name}' with similarity {sim:.4f}")
                 # else:
                 #     # Stop adding tables once similarity drops below threshold
                 #     break

            # Optional: Limit the number of returned tables to avoid overly complex prompts for LLM
            # max_relevant_tables = 5
            # relevant_tables = relevant_tables[:max_relevant_tables]

            logger.info(f"Semantic search identified relevant tables: {relevant_tables}")
            return relevant_tables

        except KeyError as e:
            logger.error(f"Schema format error during semantic search: Missing key {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            return []

    def _llm_generate_sql(self, query: str, database_schema: Dict[str, Any], relevant_tables: List[str]) -> Optional[str]:
        """
        Generates an SQL query using the loaded LLM pipeline based on the NL query and relevant schema.

        Args:
            query (str): The natural language query.
            database_schema (Dict[str, Any]): The full database schema.
            relevant_tables (List[str]): List of table names identified as relevant.

        Returns:
            Optional[str]: The generated SQL query string, or None on failure.
        """
        self._load_llm_pipeline() # Ensure LLM is loaded
        if self.llm_pipeline is None:
             raise RuntimeError("LLM pipeline is not loaded.")

        try:
            # Construct a precise prompt for the LLM
            # Include table names and their columns (with types if possible) for relevant tables
            schema_description_parts = []
            for table_name in relevant_tables:
                 if table_name in database_schema["tables"]:
                     table_info = database_schema["tables"][table_name]
                     columns = table_info.get("columns", [])
                     types = table_info.get("types", {}) # Get types if available
                     col_defs = [f"{col} ({types.get(col, 'UNKNOWN')})" for col in columns] # Include types
                     schema_description_parts.append(f"Table `{table_name}` columns: {', '.join(col_defs)}")
                 else:
                      logger.warning(f"Relevant table '{table_name}' not found in provided schema for LLM prompt.")

            if not schema_description_parts:
                 logger.error("Could not construct schema description for LLM prompt.")
                 return None

            schema_str = "\n".join(schema_description_parts)

            # Improved Prompt Engineering (adjust based on the specific LLM's strengths/weaknesses)
            # Using a format similar to common Text-to-SQL datasets can help.
            prompt = f"""Given the following SQLite database schema:
{schema_str}

Generate a SQLite SQL query that answers the question: "{query}"
Provide only the SQL query. Do not include explanations or markdown formatting.
SQL Query:"""

            logger.debug(f"LLM Prompt:\n{prompt}")

            # Generate SQL using the LLM pipeline
            # Adjust generation parameters as needed:
            # - max_length: Controls max SQL query length
            # - num_beams: Higher might find better results but slower
            # - early_stopping: Can stop generation early if EOS token is found
            # - temperature: Lower value makes output more deterministic (good for SQL)
            # - do_sample=False: Usually better for deterministic tasks like SQL generation
            generation_params = {
                 "max_length": self.DEFAULT_LLM_MAX_LENGTH,
                 "num_return_sequences": 1,
                 "do_sample": False,
                 "num_beams": 4, # Use beam search for potentially better quality
                 "early_stopping": True,
                 "temperature": 0.5, # Low temperature for less randomness
            }
            result = self.llm_pipeline(prompt, **generation_params)

            if result and isinstance(result, list) and 'generated_text' in result[0]:
                raw_sql = result[0]['generated_text'].strip()
                # Post-process: remove potential markdown, leading/trailing quotes, etc.
                clean_sql = re.sub(r"^```sql\s*|\s*```$", "", raw_sql).strip()
                # Sometimes models add extra phrases, try to extract just the SQL part
                if not clean_sql.upper().startswith("SELECT"):
                    # Try a simple regex to find SELECT statements if the model added prefix text
                    match = re.search(r"SELECT\s+.*?;?", clean_sql, re.IGNORECASE | re.DOTALL)
                    if match:
                        clean_sql = match.group(0).strip()

                logger.info(f"LLM generated SQL: '{clean_sql}' (Raw: '{raw_sql}')")
                return clean_sql
            else:
                logger.error(f"LLM pipeline returned unexpected result format: {result}")
                return None

        except Exception as e:
            logger.error(f"Error during LLM SQL generation: {e}", exc_info=True)
            return None # Return None on error

    def _validate_generated_sql(self, sql_query: Optional[str]) -> Optional[str]:
        """
        Performs basic validation checks on the generated SQL query.
        Focuses on preventing obviously incorrect or harmful queries.

        Args:
            sql_query (Optional[str]): The SQL query generated by the LLM.

        Returns:
            Optional[str]: The validated SQL query if it passes checks, otherwise None.
        """
        if not sql_query or not isinstance(sql_query, str):
            return None

        # 1. Check if it's empty
        if not sql_query.strip():
             logger.warning("Validation failed: Generated SQL is empty.")
             return None

        # 2. Basic Syntax Check (ensure it starts with SELECT)
        #    This prevents accidental generation of UPDATE, DELETE, DROP etc.
        #    (More robust parsing is complex and might require external libraries)
        normalized_sql = sql_query.strip().upper()
        if not normalized_sql.startswith("SELECT"):
            logger.warning(f"Validation failed: SQL does not start with SELECT: '{sql_query}'")
            return None

        # 3. Check for common SQL injection patterns (very basic)
        #    This is NOT foolproof protection but can catch simple cases.
        #    Real protection comes from parameterized queries where possible,
        #    or strict validation/sandboxing. LLM output is inherently risky.
        # if re.search(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s+", normalized_sql, re.IGNORECASE):
        #      logger.warning(f"Validation failed: SQL contains potentially harmful keywords after semicolon: '{sql_query}'")
        #      return None

        # 4. Check for severely unbalanced parentheses (simple heuristic)
        if sql_query.count('(') != sql_query.count(')'):
             logger.warning(f"Validation failed: SQL has unbalanced parentheses: '{sql_query}'")
             return None

        # If all checks pass, return the original (non-uppercased) query
        logger.debug(f"SQL validation passed for: '{sql_query}'")
        return sql_query.strip()


    def generate_natural_language_response(self, original_query: str, query_results: List[Dict[str, Any]]) -> str:
        """
        Generates a concise natural language summary based on the SQL query results.
        (Currently basic, could be enhanced with another LLM call for summarization).

        Args:
            original_query (str): The user's original natural language question.
            query_results (List[Dict[str, Any]]): The data returned by the SQL query (list of dicts).

        Returns:
            str: A natural language response summarizing the results.
        """
        try:
            if query_results is None: # Check for None explicitly
                 return "The query execution failed or returned an unexpected result."
            if not query_results:
                return "I couldn't find any data matching your request."

            num_rows = len(query_results)
            num_cols = len(query_results[0]) if num_rows > 0 else 0

            # Simple Summarization Logic
            if num_rows == 1:
                if num_cols == 1:
                    # Single value result
                    value = list(query_results[0].values())[0]
                    return f"The result is: **{value}**"
                else:
                    # Single row, multiple columns
                    # Example: "For the single record found, the details are: col1=val1, col2=val2..."
                    details = ", ".join([f"`{k}` = {v}" for k, v in query_results[0].items()])
                    return f"I found one record matching your request: {details}"
            else:
                # Multiple rows
                if num_cols == 1:
                    # Multiple rows, single column - List the first few values
                    col_name = list(query_results[0].keys())[0]
                    preview_count = min(num_rows, 5)
                    values_preview = ", ".join([f"'{str(row[col_name])}'" for row in query_results[:preview_count]])
                    more = "..." if num_rows > preview_count else ""
                    return f"Found {num_rows} results for `{col_name}`. The first few are: {values_preview}{more}"
                else:
                    # Multiple rows, multiple columns - General summary
                    preview_count = min(num_rows, 3) # Show first 3 rows max in summary
                    col_names = list(query_results[0].keys())
                    response = f"I found {num_rows} records matching your request. Here are the first {preview_count} rows:\n\n"
                    # Basic table format in markdown
                    response += f"| {' | '.join(col_names)} |\n"
                    response += f"| {' | '.join(['---'] * len(col_names))} |\n"
                    for row in query_results[:preview_count]:
                         response += f"| {' | '.join([str(v) for v in row.values()])} |\n"
                    if num_rows > preview_count:
                         response += f"\n... ({num_rows - preview_count} more rows available in the Raw Data tab)"

                    return response

        except Exception as e:
            logger.error(f"Error generating natural language response: {e}", exc_info=True)
            # Fallback to a generic message
            if query_results:
                 return f"I found {len(query_results)} results. Please see the Raw Data tab for details."
            else:
                 return "An error occurred while summarizing the results."
# --- END OF FILE nlp_engine.py ---