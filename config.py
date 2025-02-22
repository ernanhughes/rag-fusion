import toml
import os


class Config:
    """Configuration class that loads from `config.toml` if available, otherwise uses defaults."""

    DEFAULTS = {
        "CHAT_MODEL": "qwen2.5",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "EMBEDDING_MODEL": "mxbai-embed-large",
        "EMBEDDING_DIMS": 1024,
        "LOG_FILE": "app.log",
        "LOG_LEVEL": "DEBUG",
        "DATA_DIR": "./data",
        "DB_NAME": "rag_fusion.db",
        "SCHEMA_FILE": "schema.sql",
        "VECTOR_DB_NAME": "rag_fusion_vector.db",
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 50,
        "top_k": 5,
        "rrf_k": 60,
        "max_search_results": 5,    
        "num_queries": 5
    }

    @classmethod
    def load_config(cls, filename="config.toml"):
        """Loads configuration from a TOML file or falls back to defaults."""
        if os.path.exists(filename):
            try:
                loaded_config = toml.load(filename)
                return {**cls.DEFAULTS, **loaded_config}  # Merge defaults with loaded values
            except Exception as e:
                print(f"⚠️ Error loading TOML file: {e}. Using defaults.")
        return cls.DEFAULTS  # If file is missing or fails to load

# Load configuration
appConfig = Config.load_config()

