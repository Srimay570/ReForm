from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("No DATABASE_URL set for SQLAlchemy database")
    raise ValueError("No DATABASE_URL set for SQLAlchemy database")

logger.info(f"Using database URL: {DATABASE_URL}")

# Create database engine
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Error creating database engine: {e}")
    raise

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    habits = relationship("Habit", back_populates="user")
    health_metrics = relationship("HealthMetric", back_populates="user")
    study_logs = relationship("StudyLog", back_populates="user")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class WaitlistEntry(Base):
    __tablename__ = "waitlist_entries"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    interests = Column(String)  # Stored as comma-separated values
    created_at = Column(DateTime, default=datetime.utcnow)

class Habit(Base):
    __tablename__ = "habits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    habit = Column(String)
    category = Column(String)
    frequency = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="habits")

class HealthMetric(Base):
    __tablename__ = "health_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime)
    sleep = Column(Float)
    exercise = Column(Float)
    water = Column(Integer)
    stress = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="health_metrics")

class StudyLog(Base):
    __tablename__ = "study_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    subject = Column(String)
    duration = Column(Integer)  # in minutes
    productivity = Column(Integer)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="study_logs")

# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("All tables created successfully")
except Exception as e:
    logger.error(f"Error creating tables: {e}")
    raise

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()