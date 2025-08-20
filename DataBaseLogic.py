from sqlalchemy import Column, String, Float, Integer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from DataBaseConnection import create_db_engine, create_db_session

Base = declarative_base()


class ScanResult(Base):
    __tablename__ = 'scan_results'

    id = Column(Integer, primary_key=True, autoincrement=True)

    url = Column(String(2083), nullable=False, unique=True, index=True)

    vote_percentage = Column(Float, nullable=False)
    avg_base_confidence = Column(Float, nullable=False)
    meta_confidence = Column(Float, nullable=True)
    label = Column(String(50), nullable=False)

    def __repr__(self):
        return f"<ScanResult(url='{self.url}', label='{self.label}')>"


def init_db():
    engine = create_db_engine()
    Base.metadata.create_all(engine)


class DatabaseManager:
    def __init__(self):
        self.Session = create_db_session()

    def add_scan_result(self, url, vote_percentage, avg_base_confidence, meta_confidence, label):
        session = self.Session()
        try:
            new_result = ScanResult(
                url=url,
                vote_percentage=float(vote_percentage),
                avg_base_confidence=float(avg_base_confidence),
                meta_confidence=float(meta_confidence) if meta_confidence is not None else None,
                label=label
            )
            session.add(new_result)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            print(f"[INFO] URL already exists (race condition handled): {url}")
            return False
        except Exception as e:
            session.rollback()
            print(f"[DB ERROR] Failed to insert {url}: {e}")
            return False
        finally:
            session.close()

    def get_scan_result(self, url):
        session = self.Session()
        try:
            result = session.query(ScanResult).filter_by(url=url).first()
            return result
        finally:
            session.close()
