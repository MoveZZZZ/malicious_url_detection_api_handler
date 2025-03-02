import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def get_connection_string():
    return os.environ.get('DATABASE_CONNECTION_STRING')

def create_db_engine():
    connection_string = get_connection_string()
    engine = create_engine(connection_string, echo=False)
    return engine

def create_db_session():
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    return Session