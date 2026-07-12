from dataclasses import dataclass
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

URI_TEMPLATES={
    "PostgreSQL": "postgresql+psycopg2://{user}:{password}@{host}/{database}",
    "MySQL": "mysql+mysqlconnector://{user}:{password}@{host}/{database}"
}

@dataclass
class DBConfig:
    """Connection details, gathered by whatever UI is in front."""
    host:str
    user:str
    password:str
    database:str
    dialect:str="PostgreSQL"

    def is_complete(self) -> bool:
        return all([self.host, self.user, self.password, self.database])
    
    def uri(self) -> str:
        template=URI_TEMPLATES[self.dialect]
        return template.format(
            user=self.user,
            password=self.password,
            host=self.host,
            database=self.database
        )

def get_database(config:DBConfig)->SQLDatabase:
    """Build a LangChain SQLDatabase from a DBConfig.

    Raises:
        ValueError: if connection details are missing.
    """
    if not config.is_complete():
        raise ValueError("Please provide all DB connection details.")
    
    engine=create_engine(config.uri())
    return SQLDatabase(engine)