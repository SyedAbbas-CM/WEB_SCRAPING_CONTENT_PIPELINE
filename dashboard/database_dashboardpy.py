# dashboard/enhanced_dashboard.py
"""
Enhanced Web Dashboard with full database integration
Real-time monitoring, analytics, and control
"""

import os
import json
import time
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from collections import defaultdict
import threading

class EnhancedDashboard:
    """
    Advanced dashboard with:
    - Real-time node monitoring
    - Pipeline visualization
    - Analytics and reporting
    - Task queue management
    - Content database browser
    """
    
    def __init__(self, redis_client, db_path='data/scrapehive.db'):
        self.redis = redis_client
        self.db_path = db_path
        self.init_database()
        
        # Flask setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'scrapehive-dashboard')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio()
        
        # Start background tasks
        self.start_background_tasks()
        
    def init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        
        # Enhanced schema
        conn.executescript('''
            -- Nodes table
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                hostname TEXT,
                ip_address TEXT,
                capabilities TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            );
            
            -- Task history with analytics
            CREATE TABLE IF NOT EXISTS task_history (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                target TEXT,
                platform TEXT,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                node_id TEXT,
                execution_time REAL,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                result_size INTEGER,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );
            
            -- Scraped content with analysis
            CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                platform TEXT,
                title TEXT,
                content_text TEXT,
                author TEXT,
                created_utc TIMESTAMP,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scraped_by TEXT,
                viral_score REAL,
                sentiment TEXT,
                emotions TEXT,
                topics TEXT,
                analyzed BOOLEAN DEFAULT 0,
                script_generated BOOLEAN DEFAULT 0,
                published BOOLEAN DEFAULT 0,
                FOREIGN KEY (scraped_by) REFERENCES nodes(node_id)
            );
            
            -- Pipeline executions
            CREATE TABLE IF NOT EXISTS pipeline_executions (
                execution_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                stages_completed INTEGER DEFAULT 0,
                stages_total INTEGER,
                error_message TEXT
            );
            
            -- Performance metrics
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                node_id TEXT,
                metric_type TEXT,
                metric_value REAL,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );
            
            -- System events log
            CREATE TABLE IF NOT EXISTS system_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                severity TEXT,
                node_id TEXT,
                message TEXT,
                details TEXT
            );
            
            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_task_created ON task_history(created_at);
            CREATE INDEX IF NOT EXISTS idx_content_platform ON content(platform);
            CREATE INDEX IF NOT EXISTS idx_content_viral ON content(viral_score);
            CREATE INDEX IF NOT EXISTS idx_metrics_node ON performance_metrics(node_id, timestamp);
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_routes(self):
        """Setup all dashboard routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(ENHANCED_DASHBOARD_HTML)
        
        # API Routes
        @self.app.route('/api/dashboard/summary')
        def dashboard_summary():
            """Get dashboard summary data"""
            summary = {
                'nodes': self.get_nodes_summary(),
                'tasks': self.get_tasks_summary(),
                'content': self.get_content_summary(),
                'performance': self.get_performance_summary()
            }
            return jsonify(summary)
        
        @self.app.route('/api/nodes/detailed')
        def nodes_detailed():
            """Get detailed node information"""
            nodes = self.get_nodes_detailed()
            return jsonify(nodes)
        
        @self.app.route('/api/tasks/recent')
        def recent_tasks():
            """Get recent tasks with pagination"""
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            
            tasks = self.get_recent_tasks(page, per_page)
            return jsonify(tasks)
        
        @self.app.route('/api/content/viral')
        def viral_content():
            """Get high viral score content"""
            min_score = request.args.get('min_score', 70, type=float)
            content = self.get_viral_content(min_score)
            return jsonify(content)
        
        @self.app.route('/api/analytics/timeline')
        def analytics_timeline():
            """Get timeline analytics data"""
            timeframe = request.args.get('timeframe', '24h')
            data = self.get_timeline_analytics(timeframe)
            return jsonify(data)
        
        @self.app.route('/api/pipelines/active')
        def active_pipelines():
            """Get active pipeline executions"""
            pipelines = self.get_active_pipelines()
            return jsonify(pipelines)
        
        @self.app.route('/api/export/content')
        def export_content():
            """Export content data as CSV"""
            platform = request.args.get('platform')
            date_from = request.args.get('from')
            date_to = request.args.get('to')
            
            csv_path = self.export_content_csv(platform, date_from, date_to)
            return send_file(csv_path, as_attachment=True)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get system alerts"""
            alerts = self.get_system_alerts()
            return jsonify(alerts)
    
    def setup_socketio(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Send initial data on connect"""
            emit('dashboard_data', {
                'nodes': self.get_nodes_summary(),
                'queues': self.get_queue_status(),
                'metrics': self.get_realtime_metrics()
            })
        
        @self.socketio.on('request_node_details')
        def handle_node_details(data):
            """Send detailed node information"""
            node_id = data.get('node_id')
            details = self.get_node_details(node_id)
            emit('node_details', details)
        
        @self.socketio.on('create_manual_task')
        def handle_manual_task(data):
            """Create manual task"""
            task_id = self.create_manual_task(data)
            emit('task_created', {'task_id': task_id})
    
    def start_background_tasks(self):
        """Start background monitoring tasks"""
        
        def monitor_loop():
            while True:
                try:
                    # Collect metrics
                    metrics = self.collect_system_metrics()
                    
                    # Store in database
                    self.store_metrics(metrics)
                    
                    # Check for alerts
                    alerts = self.check_alerts(metrics)
                    
                    # Broadcast updates
                    self.socketio.emit('metrics_update', {
                        'metrics': metrics,
                        'alerts': alerts,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
                
                time.sleep(5)
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def get_nodes_summary(self) -> Dict:
        """Get summary of all nodes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get node counts by type
        cursor.execute('''
            SELECT node_type, COUNT(*) as count,
                   SUM(CASE WHEN last_seen > datetime('now', '-1 minute') THEN 1 ELSE 0 END) as online
            FROM nodes
            GROUP BY node_type
        ''')
        
        node_stats = {}
        for row in cursor.fetchall():
            node_stats[row[0]] = {
                'total': row[1],
                'online': row[2]
            }
        
        # Get current performance
        cursor.execute('''
            SELECT AVG(metric_value) as avg_cpu
            FROM performance_metrics
            WHERE metric_type = 'cpu_percent'
            AND timestamp > datetime('now', '-5 minutes')
        ''')
        
        avg_cpu = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'stats': node_stats,
            'total_online': sum(s['online'] for s in node_stats.values()),
            'avg_cpu': avg_cpu
        }
    
    def get_tasks_summary(self) -> Dict:
        """Get tasks summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Task statistics for last 24 hours
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(CASE WHEN status = 'completed' THEN execution_time ELSE NULL END) as avg_time
            FROM task_history
            WHERE created_at > datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        
        # Tasks by platform
        cursor.execute('''
            SELECT platform, COUNT(*) as count
            FROM task_history
            WHERE created_at > datetime('now', '-24 hours')
            GROUP BY platform
        ''')
        
        by_platform = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'last_24h': {
                'total': stats[0] or 0,
                'completed': stats[1] or 0,
                'failed': stats[2] or 0,
                'avg_execution_time': stats[3] or 0
            },
            'by_platform': by_platform,
            'success_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
        }
    
    def get_content_summary(self) -> Dict:
        """Get content summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Content statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT platform) as platforms,
                AVG(viral_score) as avg_viral_score,
                SUM(CASE WHEN viral_score > 75 THEN 1 ELSE 0 END) as high_viral,
                SUM(CASE WHEN analyzed = 1 THEN 1 ELSE 0 END) as analyzed,
                SUM(CASE WHEN script_generated = 1 THEN 1 ELSE 0 END) as scripts
            FROM content
            WHERE scraped_at > datetime('now', '-7 days')
        ''')
        
        stats = cursor.fetchone()
        
        # Top topics
        cursor.execute('''
            SELECT topics, COUNT(*) as count
            FROM content
            WHERE topics IS NOT NULL
            AND scraped_at > datetime('now', '-24 hours')
            GROUP BY topics
            ORDER BY count DESC
            LIMIT 10
        ''')
        
        top_topics = [(row[0], row[1]) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_content': stats[0] or 0,
            'platforms': stats[1] or 0,
            'avg_viral_score': stats[2] or 0,
            'high_viral_count': stats[3] or 0,
            'analyzed_count': stats[4] or 0,
            'scripts_generated': stats[5] or 0,
            'top_topics': top_topics
        }
    
    def get_timeline_analytics(self, timeframe: str) -> Dict:
        """Get timeline analytics data for charts"""
        conn = sqlite3.connect(self.db_path)
        
        # Determine time range
        if timeframe == '1h':
            interval = 'datetime("now", "-1 hour")'
            group_by = "strftime('%Y-%m-%d %H:%M', created_at)"
        elif timeframe == '24h':
            interval = 'datetime("now", "-24 hours")'
            group_by = "strftime('%Y-%m-%d %H', created_at)"
        else:  # 7d
            interval = 'datetime("now", "-7 days")'
            group_by = "strftime('%Y-%m-%d', created_at)"
        
        # Tasks over time
        df_tasks = pd.read_sql_query(f'''
            SELECT {group_by} as time,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
            FROM task_history
            WHERE created_at > {interval}
            GROUP BY time
            ORDER BY time
        ''', conn)
        
        # Content viral scores over time
        df_viral = pd.read_sql_query(f'''
            SELECT {group_by.replace('created_at', 'scraped_at')} as time,
                   AVG(viral_score) as avg_score,
                   MAX(viral_score) as max_score
            FROM content
            WHERE scraped_at > {interval}
            AND viral_score IS NOT NULL
            GROUP BY time
            ORDER BY time
        ''', conn)
        
        conn.close()
        
        # Create Plotly charts
        tasks_chart = go.Figure()
        tasks_chart.add_trace(go.Scatter(
            x=df_tasks['time'],
            y=df_tasks['total'],
            name='Total Tasks',
            mode='lines+markers'
        ))
        tasks_chart.add_trace(go.Scatter(
            x=df_tasks['time'],
            y=df_tasks['completed'],
            name='Completed',
            mode='lines+markers'
        ))
        
        viral_chart = go.Figure()
        viral_chart.add_trace(go.Scatter(
            x=df_viral['time'],
            y=df_viral['avg_score'],
            name='Average Viral Score',
            mode='lines+markers'
        ))
        viral_chart.add_trace(go.Scatter(
            x=df_viral['time'],
            y=df_viral['max_score'],
            name='Max Viral Score',
            mode='lines+markers'
        ))
        
        return {
            'tasks_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(tasks_chart)),
            'viral_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(viral_chart))
        }
    
    def get_viral_content(self, min_score: float) -> List[Dict]:
        """Get content with high viral scores"""
        conn = sqlite3.connect