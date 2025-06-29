#!/usr/bin/env python3
"""
Database Models and Initialization
"""

import logging
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

class DetectionRecord(Base):
    """Model for storing detection records."""
    __tablename__ = "detection_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    src_ip = Column(String(45), index=True)  # IPv6 compatible
    src_mac = Column(String(17), index=True)
    dst_ip = Column(String(45))
    dst_mac = Column(String(17))
    arp_op = Column(Integer)
    threat_level = Column(String(10))
    rule_detection = Column(Boolean)
    rule_reason = Column(Text)
    ml_prediction = Column(String(50))
    ml_confidence = Column(Float)
    combined_threat = Column(Boolean)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'src_ip': self.src_ip,
            'src_mac': self.src_mac,
            'dst_ip': self.dst_ip,
            'dst_mac': self.dst_mac,
            'arp_op': self.arp_op,
            'threat_level': self.threat_level,
            'rule_detection': self.rule_detection,
            'rule_reason': self.rule_reason,
            'ml_prediction': self.ml_prediction,
            'ml_confidence': self.ml_confidence,
            'combined_threat': self.combined_threat
        }

class AlertRecord(Base):
    """Model for storing alert records."""
    __tablename__ = "alert_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    detection_id = Column(Integer, index=True)
    alert_type = Column(String(50))  # email, webhook, etc.
    sent = Column(Boolean, default=False)
    error = Column(Text)
    recipient = Column(String(255))
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'detection_id': self.detection_id,
            'alert_type': self.alert_type,
            'sent': self.sent,
            'error': self.error,
            'recipient': self.recipient
        }

class NetworkInterface(Base):
    """Model for storing network interface information."""
    __tablename__ = "network_interfaces"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    description = Column(Text)
    ip_address = Column(String(45))
    mac_address = Column(String(17))
    is_active = Column(Boolean, default=False)
    last_used = Column(DateTime)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'ip_address': self.ip_address,
            'mac_address': self.mac_address,
            'is_active': self.is_active,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

class Configuration(Base):
    """Model for storing configuration snapshots."""
    __tablename__ = "configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    config_type = Column(String(50))  # detection, alerts, web, etc.
    config_data = Column(Text)  # JSON string
    user = Column(String(100))
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'config_type': self.config_type,
            'config_data': self.config_data,
            'user': self.user
        }

# Database connection
engine = None
SessionLocal = None

async def init_db(database_url: str = "sqlite:///data/arp_detector.db"):
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    try:
        # Create engine
        engine = create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        logger.info(f"Database initialized: {database_url}")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def get_db() -> Session:
    """Get database session."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database operations
class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        self.db = None
    
    def get_session(self) -> Session:
        """Get database session."""
        if SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return SessionLocal()
    
    async def save_detection(self, detection_data: dict) -> Optional[int]:
        """Save detection record to database."""
        try:
            db = self.get_session()
            record = DetectionRecord(**detection_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record.id
        except Exception as e:
            logger.error(f"Error saving detection: {e}")
            return None
        finally:
            db.close()
    
    async def save_alert(self, alert_data: dict) -> Optional[int]:
        """Save alert record to database."""
        try:
            db = self.get_session()
            record = AlertRecord(**alert_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record.id
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return None
        finally:
            db.close()
    
    async def get_detections(self, limit: int = 100, offset: int = 0) -> list:
        """Get detection records from database."""
        try:
            db = self.get_session()
            records = db.query(DetectionRecord).order_by(
                DetectionRecord.timestamp.desc()
            ).offset(offset).limit(limit).all()
            
            return [record.to_dict() for record in records]
        except Exception as e:
            logger.error(f"Error getting detections: {e}")
            return []
        finally:
            db.close()
    
    async def get_alerts(self, limit: int = 100, offset: int = 0) -> list:
        """Get alert records from database."""
        try:
            db = self.get_session()
            records = db.query(AlertRecord).order_by(
                AlertRecord.timestamp.desc()
            ).offset(offset).limit(limit).all()
            
            return [record.to_dict() for record in records]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
        finally:
            db.close()
    
    async def get_detection_stats(self) -> dict:
        """Get detection statistics from database."""
        try:
            db = self.get_session()
            
            total_detections = db.query(DetectionRecord).count()
            total_alerts = db.query(AlertRecord).count()
            successful_alerts = db.query(AlertRecord).filter(AlertRecord.sent == True).count()
            
            return {
                'total_detections': total_detections,
                'total_alerts': total_alerts,
                'successful_alerts': successful_alerts,
                'alert_success_rate': (successful_alerts / total_alerts * 100) if total_alerts > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return {}
        finally:
            db.close() 