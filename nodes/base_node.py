# nodes/base_node.py
"""
Base node class that all node types inherit from
Handles common functionality like heartbeat, metrics, logging
"""

import os
import sys
import json
import time
import redis
import logging
import psutil
import platform
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import colorlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NodeType(Enum):
    AI_NODE = "ai_node"
    SCRAPING_NODE = "scraping_node"
    SCHEDULER_NODE = "scheduler_node"
    VIEW_MASTER = "view_master"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    type: str
    target: str
    priority: int
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_to: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['status'] = TaskStatus(data['status'])
        return cls(**data)

class BaseNode(ABC):
    """Base class for all node types"""
    
    def __init__(self, node_id: str, node_type: NodeType, master_ip: str = None):
        self.node_id = node_id
        self.node_type = node_type
        self.master_ip = master_ip or os.getenv('MASTER_IP', 'localhost')
        
        # System info
        self.platform = platform.system()
        self.hostname = platform.node()
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        
        # Setup components
        self.logger = self._setup_logger()
        self.redis = self._setup_redis()
        
        # State
        self.running = True
        self.current_task = None
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_runtime': 0,
            'start_time': time.time()
        }
        
        # Start heartbeat
        self._start_heartbeat()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup colored logging"""
        # Create logger
        logger = logging.getLogger(self.node_id)
        logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
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
        )
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = os.path.expanduser(os.getenv('LOG_PATH', '~/.scrapehive/logs'))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{self.node_id}_{datetime.now().strftime("%Y%m%d")}.log')
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        try:
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', self.master_ip),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD'),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            r.ping()
            self.logger.info(f"Connected to Redis at {self.master_ip}")
            return r
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _start_heartbeat(self):
        """Start heartbeat thread"""
        def heartbeat_loop():
            while self.running:
                try:
                    self.send_heartbeat()
                    time.sleep(10)
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
    
    def send_heartbeat(self):
        """Send heartbeat to master"""
        info = self.get_node_info()
        
        # Store in Redis
        self.redis.hset(f'node:{self.node_id}:info', mapping=info)
        self.redis.set(f'node:{self.node_id}:heartbeat', time.time())
        self.redis.expire(f'node:{self.node_id}:heartbeat', 60)  # 1 minute TTL
        
        # Store current task if any
        if self.current_task:
            self.redis.set(
                f'node:{self.node_id}:current_task',
                json.dumps(self.current_task.to_dict())
            )
        else:
            self.redis.delete(f'node:{self.node_id}:current_task')
    
    def get_node_info(self) -> Dict:
        """Get current node information"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'platform': self.platform,
            'hostname': self.hostname,
            'status': 'running' if self.running else 'stopped',
            'cpu_count': self.cpu_count,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total': self.memory_total,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'metrics': self.metrics,
            'uptime': time.time() - self.metrics['start_time']
        }
    
    def get_task_queue(self) -> str:
        """Get the appropriate task queue name for this node"""
        if self.node_type == NodeType.AI_NODE:
            return f"ai_tasks:{self.node_id}"
        elif self.node_type == NodeType.SCRAPING_NODE:
            return f"scraping_tasks:{self.node_id}"
        elif self.node_type == NodeType.SCHEDULER_NODE:
            return "scheduling_tasks"
        else:
            return f"tasks:{self.node_id}"
    
    def get_next_task(self) -> Optional[Task]:
        """Get next task from queue"""
        queue_name = self.get_task_queue()
        
        # Check for priority tasks first
        priority_task = self.redis.rpop(f"priority_{queue_name}")
        if priority_task:
            return Task.from_dict(json.loads(priority_task))
        
        # Regular queue
        task_json = self.redis.rpop(queue_name)
        if task_json:
            return Task.from_dict(json.loads(task_json))
        
        return None
    
    def report_task_completion(self, task: Task):
        """Report task completion to master"""
        # Update metrics
        if task.status == TaskStatus.COMPLETED:
            self.metrics['tasks_completed'] += 1
        else:
            self.metrics['tasks_failed'] += 1
        
        # Store result
        self.redis.hset(f"task_result:{task.id}", mapping={
            'status': task.status.value,
            'completed_at': task.completed_at,
            'node_id': self.node_id,
            'result': json.dumps(task.result) if task.result else None,
            'error': task.error
        })
        
        # Notify completion
        self.redis.publish('task_completions', json.dumps({
            'task_id': task.id,
            'status': task.status.value,
            'node_id': self.node_id
        }))
    
    @abstractmethod
    def process_task(self, task: Task) -> Dict:
        """Process a task - must be implemented by subclasses"""
        pass
    
    def run(self):
        """Main node loop"""
        self.logger.info(f"{self.node_type.value} '{self.node_id}' started")
        self.logger.info(f"Platform: {self.platform}, CPUs: {self.cpu_count}, Memory: {self.memory_total / (1024**3):.1f} GB")
        
        try:
            while self.running:
                try:
                    # Get next task
                    task = self.get_next_task()
                    
                    if task:
                        self.logger.info(f"Processing task {task.id}: {task.type}")
                        self.current_task = task
                        
                        # Process task
                        task.started_at = time.time()
                        task.status = TaskStatus.RUNNING
                        task.assigned_to = self.node_id
                        
                        try:
                            result = self.process_task(task)
                            task.result = result
                            task.status = TaskStatus.COMPLETED
                            self.logger.info(f"Task {task.id} completed successfully")
                        except Exception as e:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            self.logger.error(f"Task {task.id} failed: {e}")
                        
                        task.completed_at = time.time()
                        self.report_task_completion(task)
                        self.current_task = None
                    else:
                        # No tasks, wait
                        time.sleep(2)
                        
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.logger.info(f"Node {self.node_id} shutdown complete")