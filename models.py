from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class RequestLog(Base):
    """Model for logging API requests"""
    __tablename__ = 'request_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String, index=True)
    client_ip = Column(String)
    request_body = Column(Text)  # Store JSON as text
    
    # Relationship to ResponseLog
    response = relationship("ResponseLog", back_populates="request", uselist=False)

class ResponseLog(Base):
    """Model for logging API responses"""
    __tablename__ = 'response_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey('request_logs.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Float)
    response = Column(Text)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    model_used = Column(String)  # Track which model was used
    
    # Relationship back to RequestLog
    request = relationship("RequestLog", back_populates="response")