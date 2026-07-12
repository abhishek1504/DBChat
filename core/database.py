from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

URI_TEMPLATES = {
    "PostgreSQL": "postgresql+psycopg2://{user}:{password}@{host}/{database}",
    "MySQL": "mysql+mysqlconnector://{user}:{password}@{host}/{database}",
}

# Best-effort, dialect-specific ways to tell the DB driver itself the
# connection is read-only, on top of the SQL-level guard in
# core/sql_guard.py. This is a second layer, not the primary defense: the
# guard in agent.py is what actually stops a write from being attempted,
# this is what stops one from *succeeding* if it ever got past the guard.
_READONLY_EXECUTION_OPTIONS = {
    "PostgreSQL": {"postgresql_readonly": True},
}


@dataclass
class DBConfig:
    """Connection details, gathered by whatever UI is in front."""
    host: str
    user: str
    password: str
    database: str
    dialect: str = "PostgreSQL"

    def is_complete(self) -> bool:
        return all([self.host, self.user, self.password, self.database])

    def uri(self) -> str:
        template = URI_TEMPLATES[self.dialect]
        return template.format(
            user=self.user,
            password=self.password,
            host=self.host,
            database=self.database,
        )


def get_database(config: DBConfig) -> SQLDatabase:
    """Build a LangChain SQLDatabase from a DBConfig.

    Raises:
        ValueError: if connection details are missing.
    """
    if not config.is_complete():
        raise ValueError("Please provide all DB connection details.")

    engine = create_engine(config.uri())

    readonly_opts = _READONLY_EXECUTION_OPTIONS.get(config.dialect)
    if readonly_opts:
        try:
            engine = engine.execution_options(**readonly_opts)
        except Exception:
            # Best-effort: if a driver doesn't support this, the SQL-level
            # guard in core/sql_guard.py is still the real backstop.
            pass

    return SQLDatabase(engine)
