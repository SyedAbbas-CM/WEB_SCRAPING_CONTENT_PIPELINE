# nodes/view_master.py
"""
View Master Node - Web dashboard and control center
Provides monitoring, control, and database management
"""

import os
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from .base_node import BaseNode, NodeType, Task

class ViewMasterNode(BaseNode):
    """
    Master control node with:
    - Web dashboard
    - Real-time monitoring
    - Pipeline management
    - Queue control
    - Database storage
    - Analytics
    """
    
    def __init__(self, node_id: str = "view-master-01", master_ip: str = None):
        # Initialize as view master type
        super().__init__(node_id, NodeType.VIEW_MASTER, master_ip)
        
        # Flask setup
        self.app = Flask(__name__, 
                         template_folder='../dashboard/templates',
                         static_folder='../dashboard/static')
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'scrapehive-secret-key')
        CORS(self.app)
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Database setup
        self.db = self._init_database()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        # Start background threads
        self._start_monitoring()
        self._start_analytics()
        
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database"""
        db_path = os.path.expanduser(os.getenv('DATABASE_PATH', '~/.scrapehive/master.db'))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Create tables
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS pipelines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_run REAL,
                run_count INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1
            );
            
            CREATE TABLE IF NOT EXISTS task_history (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                target TEXT,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                node_id TEXT,
                result TEXT,
                error TEXT
            );
            
            CREATE TABLE IF NOT EXISTS scraped_content (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                platform TEXT,
                title TEXT,
                content TEXT,
                metadata TEXT,
                scraped_at REAL NOT NULL,
                scraped_by TEXT,
                analyzed INTEGER DEFAULT 0,
                viral_score REAL
            );
            
            CREATE TABLE IF NOT EXISTS node_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                tasks_completed INTEGER,
                tasks_failed INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                node_id TEXT,
                message TEXT,
                severity TEXT
            );
        ''')
        
        conn.commit()
        return conn
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        # API Routes
        @self.app.route('/api/nodes')
        def api_nodes():
            """Get all nodes status"""
            nodes = self._get_all_nodes()
            return jsonify(nodes)
        
        @self.app.route('/api/nodes/<node_id>')
        def api_node_detail(node_id):
            """Get specific node details"""
            node = self._get_node_details(node_id)
            return jsonify(node)
        
        @self.app.route('/api/pipelines')
        def api_pipelines():
            """Get all pipelines"""
            pipelines = self._get_pipelines()
            return jsonify(pipelines)
        
        @self.app.route('/api/pipelines', methods=['POST'])
        def api_create_pipeline():
            """Create new pipeline"""
            config = request.json
            pipeline_id = self._create_pipeline(config)
            return jsonify({'id': pipeline_id, 'status': 'created'})
        
        @self.app.route('/api/pipelines/<pipeline_id>', methods=['PUT'])
        def api_update_pipeline(pipeline_id):
            """Update pipeline"""
            updates = request.json
            self._update_pipeline(pipeline_id, updates)
            return jsonify({'status': 'updated'})
        
        @self.app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
        def api_delete_pipeline(pipeline_id):
            """Delete pipeline"""
            self._delete_pipeline(pipeline_id)
            return jsonify({'status': 'deleted'})
        
        @self.app.route('/api/pipelines/<pipeline_id>/execute', methods=['POST'])
        def api_execute_pipeline(pipeline_id):
            """Execute pipeline"""
            execution_id = self._execute_pipeline(pipeline_id)
            return jsonify({'execution_id': execution_id})
        
        @self.app.route('/api/tasks/create', methods=['POST'])
        def api_create_task():
            """Create manual task"""
            task_data = request.json
            task_id = self._create_manual_task(task_data)
            return jsonify({'task_id': task_id})
        
        @self.app.route('/api/tasks')
        def api_tasks():
            """Get task history"""
            limit = request.args.get('limit', 100, type=int)
            tasks = self._get_task_history(limit)
            return jsonify(tasks)
        
        @self.app.route('/api/queues')
        def api_queues():
            """Get all queue statuses"""
            queues = self._get_queue_status()
            return jsonify(queues)
        
        @self.app.route('/api/queues/<queue_name>/peek')
        def api_queue_peek(queue_name):
            """Peek at queue items"""
            items = self._peek_queue(queue_name)
            return jsonify(items)
        
        @self.app.route('/api/content')
        def api_content():
            """Get scraped content"""
            limit = request.args.get('limit', 100, type=int)
            platform = request.args.get('platform')
            viral_only = request.args.get('viral', False, type=bool)
            
            content = self._get_content(limit, platform, viral_only)
            return jsonify(content)
        
        @self.app.route('/api/analytics')
        def api_analytics():
            """Get system analytics"""
            analytics = self._get_analytics()
            return jsonify(analytics)
        
        @self.app.route('/api/logs/<node_id>')
        def api_logs(node_id):
            """Get node logs"""
            lines = request.args.get('lines', 100, type=int)
            logs = self._get_node_logs(node_id, lines)
            return jsonify({'logs': logs})
    
    def _setup_socketio(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Client connected"""
            self.logger.info("Dashboard client connected")
            
            # Send initial state
            emit('initial_state', {
                'nodes': self._get_all_nodes(),
                'pipelines': self._get_pipelines(),
                'queues': self._get_queue_status(),
                'analytics': self._get_analytics()
            })
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Manual update request"""
            update_type = data.get('type', 'all')
            
            if update_type == 'nodes':
                emit('nodes_update', self._get_all_nodes())
            elif update_type == 'queues':
                emit('queues_update', self._get_queue_status())
            else:
                emit('full_update', {
                    'nodes': self._get_all_nodes(),
                    'queues': self._get_queue_status()
                })
        
        @self.socketio.on('execute_pipeline')
        def handle_execute_pipeline(data):
            """Execute pipeline via WebSocket"""
            pipeline_id = data.get('pipeline_id')
            execution_id = self._execute_pipeline(pipeline_id)
            emit('pipeline_executed', {'execution_id': execution_id})
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        def monitor_loop():
            while self.running:
                try:
                    # Collect metrics
                    metrics = self._collect_metrics()
                    
                    # Store in database
                    self._store_metrics(metrics)
                    
                    # Broadcast updates
                    self.socketio.emit('metrics_update', metrics)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def _start_analytics(self):
        """Start analytics processing"""
        def analytics_loop():
            while self.running:
                try:
                    # Process analytics
                    self._process_analytics()
                    
                    time.sleep(300)  # Every 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Analytics error: {e}")
                    time.sleep(600)
        
        thread = threading.Thread(target=analytics_loop, daemon=True)
        thread.start()
    
    def _get_all_nodes(self) -> Dict:
        """Get status of all nodes"""
        nodes = {}
        
        # Get all node info from Redis
        for key in self.redis.keys('node:*:info'):
            node_id = key.split(':')[1]
            
            info = self.redis.hgetall(key)
            heartbeat = self.redis.get(f'node:{node_id}:heartbeat')
            
            if info:
                # Check if node is alive
                if heartbeat:
                    last_seen = float(heartbeat)
                    info['online'] = (time.time() - last_seen) < 60
                else:
                    info['online'] = False
                
                # Get current task
                current_task = self.redis.get(f'node:{node_id}:current_task')
                if current_task:
                    info['current_task'] = json.loads(current_task)
                
                nodes[node_id] = info
        
        return nodes
    
    def _get_pipelines(self) -> List[Dict]:
        """Get all pipelines from database"""
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT id, name, description, config, created_at, last_run, run_count, active
            FROM pipelines
            ORDER BY created_at DESC
        ''')
        
        pipelines = []
        for row in cursor.fetchall():
            pipeline = dict(row)
            pipeline['config'] = json.loads(pipeline['config'])
            pipelines.append(pipeline)
        
        return pipelines
    
    def _create_pipeline(self, config: Dict) -> str:
        """Create new pipeline"""
        pipeline_id = f"pipe_{int(time.time())}"
        
        # Store in database
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO pipelines (id, name, description, config, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pipeline_id,
            config['name'],
            config.get('description', ''),
            json.dumps(config),
            time.time()
        ))
        self.db.commit()
        
        # Queue to scheduler
        task = {
            'id': f'create_pipe_{pipeline_id}',
            'type': 'create_pipeline',
            'target': 'pipeline_creation',
            'priority': 2,
            'status': 'pending',
            'created_at': time.time(),
            'metadata': config
        }
        
        self.redis.lpush('scheduling_tasks', json.dumps(task))
        
        # Log event
        self._log_event('pipeline_created', f"Created pipeline: {config['name']}")
        
        return pipeline_id
    
    def _execute_pipeline(self, pipeline_id: str) -> str:
        """Execute pipeline"""
        execution_id = f"exec_{int(time.time())}"
        
        # Queue execution
        task = {
            'id': execution_id,
            'type': 'execute_pipeline',
            'target': 'pipeline_execution',
            'priority': 1,
            'status': 'pending',
            'created_at': time.time(),
            'metadata': {'pipeline_id': pipeline_id}
        }
        
        self.redis.lpush('scheduling_tasks', json.dumps(task))
        
        # Update last run
        cursor = self.db.cursor()
        cursor.execute('''
            UPDATE pipelines 
            SET last_run = ?, run_count = run_count + 1
            WHERE id = ?
        ''', (time.time(), pipeline_id))
        self.db.commit()
        
        return execution_id
    
    def _create_manual_task(self, task_data: Dict) -> str:
        """Create manual task"""
        task = Task(
            id=f"manual_{int(time.time())}",
            type=task_data['type'],
            target=task_data['target'],
            priority=task_data.get('priority', 1),
            status='pending',
            created_at=time.time(),
            metadata=task_data.get('metadata', {})
        )
        
        # Route to appropriate queue
        node_id = task_data.get('node_id')
        if node_id:
            # Specific node
            queue = f"priority_tasks:{node_id}"
        else:
            # Auto-route based on type
            if task.type in ['predict_virality', 'analyze_content', 'generate_script']:
                queue = 'ai_tasks:ai-node-01'
            elif task.type in ['scrape_url', 'scrape_batch']:
                queue = 'scraping_tasks:scraper-01'
            else:
                queue = 'tasks:general'
        
        # Queue task
        self.redis.lpush(queue, json.dumps(task.to_dict()))
        
        # Store in history
        self._store_task_history(task)
        
        return task.id
    
    def _get_queue_status(self) -> Dict:
        """Get status of all queues"""
        queues = {}
        
        # Define queue patterns
        patterns = [
            'ai_tasks:*',
            'scraping_tasks:*',
            'scheduling_tasks',
            'priority_tasks:*',
            'tasks:*'
        ]
        
        for pattern in patterns:
            for key in self.redis.keys(pattern):
                length = self.redis.llen(key)
                
                # Peek at first few items
                items = []
                for i in range(min(5, length)):
                    item = self.redis.lindex(key, i)
                    if item:
                        items.append(json.loads(item))
                
                queues[key] = {
                    'length': length,
                    'items': items,
                    'type': key.split(':')[0]
                }
        
        return queues
    
    def _collect_metrics(self) -> Dict:
        """Collect system metrics"""
        metrics = {
            'timestamp': time.time(),
            'nodes': {},
            'queues': {},
            'totals': {
                'nodes_online': 0,
                'total_tasks': 0,
                'tasks_per_minute': 0,
                'success_rate': 0
            }
        }
        
        # Node metrics
        nodes = self._get_all_nodes()
        for node_id, info in nodes.items():
            metrics['nodes'][node_id] = {
                'online': info.get('online', False),
                'cpu': info.get('cpu_percent', 0),
                'memory': info.get('memory_percent', 0),
                'tasks_completed': info.get('metrics', {}).get('tasks_completed', 0)
            }
            
            if info.get('online'):
                metrics['totals']['nodes_online'] += 1
        
        # Queue metrics
        queue_status = self._get_queue_status()
        for queue_name, status in queue_status.items():
            metrics['queues'][queue_name] = status['length']
            metrics['totals']['total_tasks'] += status['length']
        
        return metrics
    
    def _log_event(self, event_type: str, message: str, severity: str = 'info'):
        """Log system event"""
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO system_events (timestamp, event_type, node_id, message, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (time.time(), event_type, self.node_id, message, severity))
        self.db.commit()
    
    def process_task(self, task: Task) -> Dict:
        """View master doesn't process regular tasks"""
        return {'status': 'not_applicable'}
    
    def run(self):
        """Run the view master"""
        self.logger.info(f"View Master starting on port 5000")
        
        # Run Flask app with SocketIO
        self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False)