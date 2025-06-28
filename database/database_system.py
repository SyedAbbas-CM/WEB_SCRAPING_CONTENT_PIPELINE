# database/connection.py
"""
Database connection management
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from config import get_config

# Base class for all models
Base = declarative_base()

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, config=None):
        self.config = config or get_config().database
        self.logger = logging.getLogger('database')
        self._engine = None
        self._session_factory = None
        
    @property
    def engine(self):
        """Get or create database engine"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    @property
    def session_factory(self):
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    def _create_engine(self):
        """Create database engine with proper configuration"""
        engine_kwargs = {
            'echo': self.config.echo,
            'poolclass': QueuePool,
            'pool_size': self.config.pool_size,
            'max_overflow': self.config.max_overflow,
            'pool_pre_ping': True,  # Validate connections before use
            'pool_recycle': 3600,   # Recycle connections after 1 hour
        }
        
        # SQLite-specific settings
        if 'sqlite' in self.config.url:
            engine_kwargs.update({
                'poolclass': None,  # SQLite doesn't support connection pooling
                'connect_args': {'check_same_thread': False}
            })
        
        engine = create_engine(self.config.url, **engine_kwargs)
        
        # Add logging for slow queries
        if self.config.echo:
            @event.listens_for(engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                context._query_start_time = time.time()
            
            @event.listens_for(engine, "after_cursor_execute")
            def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                total = time.time() - context._query_start_time
                if total > 0.5:  # Log queries taking more than 500ms
                    self.logger.warning(f"Slow query ({total:.2f}s): {statement[:200]}...")
        
        return engine
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables"""
        from .models import *  # Import all models
        Base.metadata.create_all(self.engine)
        self.logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all database tables"""
        Base.metadata.drop_all(self.engine)
        self.logger.info("Database tables dropped")
    
    def health_check(self) -> bool:
        """Check database health"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def get_session() -> Session:
    """Get a new database session"""
    return get_db_manager().session_factory()

# database/models.py
"""
SQLAlchemy database models
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, 
    JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from .connection import Base

class Node(Base):
    """Node registration and status"""
    __tablename__ = 'nodes'
    
    id = Column(String(50), primary_key=True)
    node_type = Column(String(20), nullable=False)
    hostname = Column(String(100))
    ip_address = Column(String(45))  # IPv6 support
    capabilities = Column(JSON)
    status = Column(String(20), default='offline')
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relationships
    tasks = relationship("TaskHistory", back_populates="node")
    metrics = relationship("NodeMetric", back_populates="node")
    
    __table_args__ = (
        Index('idx_node_type_status', 'node_type', 'status'),
    )

class TaskHistory(Base):
    """Task execution history"""
    __tablename__ = 'task_history'
    
    id = Column(String(50), primary_key=True)
    task_type = Column(String(50), nullable=False)
    target = Column(Text)
    platform = Column(String(20))
    status = Column(String(20), nullable=False)
    priority = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Execution details
    node_id = Column(String(50), ForeignKey('nodes.id'))
    execution_time = Column(Float)  # seconds
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    result_size = Column(Integer)
    
    # JSON fields
    metadata = Column(JSON)
    result = Column(JSON)
    
    # Relationships
    node = relationship("Node", back_populates="tasks")
    
    __table_args__ = (
        Index('idx_task_created', 'created_at'),
        Index('idx_task_status', 'status'),
        Index('idx_task_platform', 'platform'),
    )
    
    @hybrid_property
    def duration(self):
        """Calculate task duration"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

class ScrapedContent(Base):
    """Scraped content storage"""
    __tablename__ = 'scraped_content'
    
    id = Column(String(50), primary_key=True)
    url = Column(Text, nullable=False)
    platform = Column(String(20))
    content_type = Column(String(20))  # post, video, image, etc.
    
    # Content fields
    title = Column(Text)
    content_text = Column(Text)
    author = Column(String(100))
    created_utc = Column(DateTime)
    
    # Scraping metadata
    scraped_at = Column(DateTime, default=datetime.utcnow)
    scraped_by = Column(String(50), ForeignKey('nodes.id'))
    scrape_method = Column(String(20))  # api, web, mobile
    
    # Analysis results
    viral_score = Column(Float)
    sentiment = Column(String(20))
    emotions = Column(JSON)
    topics = Column(JSON)
    keywords = Column(JSON)
    
    # Processing flags
    analyzed = Column(Boolean, default=False)
    script_generated = Column(Boolean, default=False)
    video_generated = Column(Boolean, default=False)
    published = Column(Boolean, default=False)
    
    # Engagement metrics
    metrics = Column(JSON)  # likes, shares, comments, etc.
    
    # Raw data
    raw_data = Column(JSON)
    
    __table_args__ = (
        Index('idx_content_platform', 'platform'),
        Index('idx_content_viral_score', 'viral_score'),
        Index('idx_content_scraped_at', 'scraped_at'),
        Index('idx_content_analyzed', 'analyzed'),
    )

class Pipeline(Base):
    """Pipeline definitions"""
    __tablename__ = 'pipelines'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    config = Column(JSON, nullable=False)
    
    # Status
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Scheduling
    schedule = Column(JSON)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    run_count = Column(Integer, default=0)
    
    # Metadata
    tags = Column(JSON)
    created_by = Column(String(50))
    
    # Relationships
    executions = relationship("PipelineExecution", back_populates="pipeline")
    
    __table_args__ = (
        Index('idx_pipeline_active', 'active'),
        Index('idx_pipeline_next_run', 'next_run'),
    )

class PipelineExecution(Base):
    """Pipeline execution history"""
    __tablename__ = 'pipeline_executions'
    
    id = Column(String(50), primary_key=True)
    pipeline_id = Column(String(50), ForeignKey('pipelines.id'), nullable=False)
    
    # Execution status
    status = Column(String(20), nullable=False)  # running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Results
    stages_completed = Column(Integer, default=0)
    stages_total = Column(Integer)
    stage_results = Column(JSON)
    error_message = Column(Text)
    
    # Context
    context = Column(JSON)
    triggered_by = Column(String(20))  # manual, schedule, event
    
    # Relationships
    pipeline = relationship("Pipeline", back_populates="executions")
    
    __table_args__ = (
        Index('idx_execution_pipeline', 'pipeline_id'),
        Index('idx_execution_status', 'status'),
        Index('idx_execution_started', 'started_at'),
    )

class NodeMetric(Base):
    """Node performance metrics"""
    __tablename__ = 'node_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(50), ForeignKey('nodes.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # System metrics
    cpu_percent = Column(Float)
    memory_percent = Column(Float)
    disk_percent = Column(Float)
    
    # Task metrics
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    avg_task_time = Column(Float)
    
    # Custom metrics
    custom_metrics = Column(JSON)
    
    # Relationships
    node = relationship("Node", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metrics_node_timestamp', 'node_id', 'timestamp'),
    )

class SystemEvent(Base):
    """System events and alerts"""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Event details
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), default='info')  # debug, info, warning, error, critical
    node_id = Column(String(50), ForeignKey('nodes.id'))
    
    # Message
    message = Column(Text, nullable=False)
    details = Column(JSON)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(String(50))
    
    __table_args__ = (
        Index('idx_events_timestamp', 'timestamp'),
        Index('idx_events_type_severity', 'event_type', 'severity'),
        Index('idx_events_resolved', 'resolved'),
    )

class User(Base):
    """User accounts for dashboard access"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    
    # Profile
    full_name = Column(String(100))
    role = Column(String(20), default='user')  # admin, user, viewer
    
    # Status
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Settings
    preferences = Column(JSON)
    
    __table_args__ = (
        Index('idx_user_username', 'username'),
        Index('idx_user_email', 'email'),
    )

class APIKey(Base):
    """API keys for external access"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String(128), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Permissions
    permissions = Column(JSON)  # List of allowed endpoints/actions
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    
    # Status
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_apikey_hash', 'key_hash'),
        Index('idx_apikey_active', 'active'),
    )

# database/migrations/001_initial_schema.py
"""
Initial database schema migration
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    """Create initial schema"""
    # Create nodes table
    op.create_table('nodes',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('node_type', sa.String(20), nullable=False),
        sa.Column('hostname', sa.String(100)),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('capabilities', sa.JSON()),
        sa.Column('status', sa.String(20), default='offline'),
        sa.Column('first_seen', sa.DateTime, default=sa.func.now()),
        sa.Column('last_seen', sa.DateTime, default=sa.func.now()),
        sa.Column('metadata', sa.JSON())
    )
    
    # Create indexes
    op.create_index('idx_node_type_status', 'nodes', ['node_type', 'status'])
    
    # Create other tables...
    # (Similar pattern for all tables)

def downgrade():
    """Drop initial schema"""
    op.drop_table('nodes')
    # Drop other tables...