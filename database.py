import sqlite3

from config import appConfig
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_name=appConfig["DB_NAME"], schema_file=appConfig["SCHEMA_FILE"]):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.load_schema(schema_file)

    def load_schema(self, schema_file):
        """Load the database schema from an external SQL file."""
        with open(schema_file, "r") as f:
            schema_sql = f.read()
        self.cursor.executescript(schema_sql)
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
