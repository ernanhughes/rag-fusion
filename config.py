import toml
import os


class Config:
    """Configuration class that loads from `config.toml` if available, otherwise uses defaults."""

    DEFAULTS = {
        "MODEL_NAME": "qwen2.5",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "EMBEDDING_MODEL_NAME": "mxbai-embed-large",
        "LOG_FILE": "app.log",
        "LOG_LEVEL": "DEBUG",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "rrf_k": 60,
        "num_queries": 3
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
config = Config.load_config()

