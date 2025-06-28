# utils/logger.py
"""
Enhanced logging system for ScrapeHive
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import colorlog
from config import get_config

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        # Create log entry
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'node_id'):
            log_entry['node_id'] = record.node_id
        
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
        
        if hasattr(record, 'platform'):
            log_entry['platform'] = record.platform
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry['stack_trace'] = record.stack_info
        
        return json.dumps(log_entry)

class RedisHandler(logging.Handler):
    """Redis handler for centralized logging"""
    
    def __init__(self, redis_client, key='logs', max_length=10000):
        super().__init__()
        self.redis = redis_client
        self.key = key
        self.max_length = max_length
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            # Push to Redis list
            self.redis.lpush(self.key, log_entry)
            
            # Trim list to max length
            self.redis.ltrim(self.key, 0, self.max_length - 1)
            
        except Exception:
            self.handleError(record)

class MetricsHandler(logging.Handler):
    """Handler that updates metrics based on log levels"""
    
    def __init__(self, metrics_collector):
        super().__init__()
        self.metrics = metrics_collector
    
    def emit(self, record):
        try:
            # Count log entries by level
            self.metrics.increment(f'logs.{record.levelname.lower()}.count')
            
            # Track errors by module
            if record.levelno >= logging.ERROR:
                self.metrics.increment(f'errors.{record.module}.count')
            
        except Exception:
            self.handleError(record)

def setup_logging(config=None, node_id=None, redis_client=None):
    """Setup logging configuration"""
    config = config or get_config()
    
    # Create logs directory
    log_dir = Path(config.logs_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'scrapehive.log',
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # JSON file handler for structured logs
    json_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'scrapehive.json',
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Redis handler for centralized logging
    if redis_client:
        redis_handler = RedisHandler(redis_client)
        redis_handler.setLevel(logging.INFO)
        redis_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(redis_handler)
    
    # Add node_id to all log records
    if node_id:
        class NodeFilter(logging.Filter):
            def filter(self, record):
                record.node_id = node_id
                return True
        
        node_filter = NodeFilter()
        for handler in root_logger.handlers:
            handler.addFilter(node_filter)
    
    # Set levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    
    return root_logger

# monitoring/metrics_collector.py
"""
System metrics collection and monitoring
"""

import time
import psutil
import threading
from typing import Dict, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import redis
import json
import logging

@dataclass
class Metric:
    """A single metric measurement"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, redis_client=None, node_id=None):
        self.redis = redis_client
        self.node_id = node_id
        self.logger = logging.getLogger('metrics_collector')
        
        # Metric storage
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        
        # Collection settings
        self.collection_interval = 60  # seconds
        self.retention_hours = 24
        
        # Custom metric collectors
        self.custom_collectors = []
        
        # Start collection thread
        self._start_collection()
    
    def increment(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.counters[name] += value
        self._store_metric(name, self.counters[name], 'counter', tags)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        self._store_metric(name, value, 'gauge', tags)
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Add value to histogram"""
        self.histograms[name].append(value)
        self._store_metric(name, value, 'histogram', tags)
    
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags)
    
    def add_collector(self, collector_func: Callable[[], Dict[str, float]]):
        """Add custom metric collector function"""
        self.custom_collectors.append(collector_func)
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['system.cpu.percent'] = cpu_percent
            metrics['system.cpu.count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['system.memory.total'] = memory.total
            metrics['system.memory.available'] = memory.available
            metrics['system.memory.percent'] = memory.percent
            metrics['system.memory.used'] = memory.used
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['system.disk.total'] = disk.total
            metrics['system.disk.used'] = disk.used
            metrics['system.disk.free'] = disk.free
            metrics['system.disk.percent'] = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['system.network.bytes_sent'] = network.bytes_sent
            metrics['system.network.bytes_recv'] = network.bytes_recv
            metrics['system.network.packets_sent'] = network.packets_sent
            metrics['system.network.packets_recv'] = network.packets_recv
            
            # Process metrics
            process = psutil.Process()
            metrics['process.cpu.percent'] = process.cpu_percent()
            metrics['process.memory.rss'] = process.memory_info().rss
            metrics['process.memory.vms'] = process.memory_info().vms
            metrics['process.threads'] = process.num_threads()
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a histogram metric"""
        values = list(self.histograms[name])
        if not values:
            return {}
        
        values.sort()
        length = len(values)
        
        return {
            'count': length,
            'min