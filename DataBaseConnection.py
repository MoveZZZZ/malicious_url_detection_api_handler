from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

_engine = None

def get_connection_string():
    return os.environ.get('DATABASE_CONNECTION_STRING')

def create_db_engine():
    global _engine
    if _engine is None:
        connection_string = get_connection_string()
        _engine = create_engine(connection_string, echo=False, pool_pre_ping=True)
    return _engine

def create_db_session():
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    return Session