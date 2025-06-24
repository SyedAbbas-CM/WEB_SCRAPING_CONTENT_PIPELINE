# nodes/scheduler_node.py
"""
Scheduler Node - Manages pipelines and schedules
Also performs light scraping when idle
"""

import os
import json
import time
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .base_node import BaseNode, NodeType, Task
from utils.pipeline_manager import Pipeline, PipelineStage

@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: str
    pipeline_id: str
    schedule_type: str  # 'once', 'recurring', 'cron'
    schedule_data: Dict
    next_run: float
    last_run: Optional[float] = None
    active: bool = True

class SchedulerNode(BaseNode):
    """
    Scheduler node that:
    - Manages content pipelines
    - Schedules recurring tasks
    - Performs light scraping when idle
    - Orchestrates multi-stage workflows
    """
    
    def __init__(self, node_id: str, master_ip: str = None):
        super().__init__(node_id, NodeType.SCHEDULER_NODE, master_ip)
        
        # Pipeline management
        self.pipelines = {}
        self.scheduled_tasks = {}
        
        # Load saved pipelines
        self._load_pipelines()
        
        # Start scheduler thread
        self._start_scheduler()
        
        # Light scraping setup
        self.can_scrape = self._check_scraping_capability()
        
        # Metrics
        self.scheduler_metrics = {
            'pipelines_created': 0,
            'pipelines_executed': 0,
            'tasks_scheduled': 0,
            'light_scrapes': 0
        }
        
    def _load_pipelines(self):
        """Load saved pipelines from Redis"""
        pipeline_keys = self.redis.keys('pipeline:*')
        
        for key in pipeline_keys:
            pipeline_data = self.redis.hgetall(key)
            if pipeline_data:
                pipeline = Pipeline.from_dict(pipeline_data)
                self.pipelines[pipeline.id] = pipeline
                
        self.logger.info(f"Loaded {len(self.pipelines)} pipelines")
    
    def _start_scheduler(self):
        """Start the scheduling thread"""
        def scheduler_loop():
            while self.running:
                schedule.run_pending()
                self._check_scheduled_tasks()
                time.sleep(30)  # Check every 30 seconds
        
        thread = threading.Thread(target=scheduler_loop, daemon=True)
        thread.start()
    
    def _check_scraping_capability(self) -> bool:
        """Check if node can do light scraping"""
        try:
            import requests
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            self.logger.warning("Light scraping not available - missing dependencies")
            return False
    
    def get_node_info(self) -> Dict:
        """Add scheduler-specific info"""
        info = super().get_node_info()
        info['pipelines_count'] = len(self.pipelines)
        info['active_schedules'] = len([t for t in self.scheduled_tasks.values() if t.active])
        info['can_scrape'] = self.can_scrape
        info['scheduler_metrics'] = self.scheduler_metrics
        return info
    
    def process_task(self, task: Task) -> Dict:
        """Process scheduler tasks"""
        task_type = task.type
        
        if task_type == 'create_pipeline':
            return self.create_pipeline(task.metadata)
        elif task_type == 'execute_pipeline':
            return self.execute_pipeline(task.metadata['pipeline_id'])
        elif task_type == 'schedule_task':
            return self.schedule_task(task.metadata)
        elif task_type == 'update_pipeline':
            return self.update_pipeline(task.metadata)
        elif task_type == 'light_scrape':
            return self.do_light_scrape(task.target, task.metadata)
        else:
            raise ValueError(f"Unknown scheduler task: {task_type}")
    
    def create_pipeline(self, config: Dict) -> Dict:
        """Create a new content pipeline"""
        # Create pipeline object
        pipeline = Pipeline(
            id=f"pipe_{int(time.time())}",
            name=config['name'],
            description=config.get('description', ''),
            stages=[],
            schedule=config.get('schedule'),
            active=True,
            created_at=time.time()
        )
        
        # Add stages
        for stage_config in config['stages']:
            stage = PipelineStage(
                id=f"stage_{len(pipeline.stages)}",
                name=stage_config['name'],
                type=stage_config['type'],
                config=stage_config.get('config', {}),
                dependencies=stage_config.get('dependencies', [])
            )
            pipeline.stages.append(stage)
        
        # Validate pipeline
        if not self._validate_pipeline(pipeline):
            raise ValueError("Invalid pipeline configuration")
        
        # Save pipeline
        self.pipelines[pipeline.id] = pipeline
        self._save_pipeline(pipeline)
        
        # Schedule if needed
        if pipeline.schedule:
            self._schedule_pipeline(pipeline)
        
        self.scheduler_metrics['pipelines_created'] += 1
        
        self.logger.info(f"Created pipeline: {pipeline.name} ({pipeline.id})")
        
        return {
            'pipeline_id': pipeline.id,
            'status': 'created',
            'stages': len(pipeline.stages)
        }
    
    def execute_pipeline(self, pipeline_id: str) -> Dict:
        """Execute a pipeline"""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        if not pipeline.active:
            raise ValueError(f"Pipeline is not active: {pipeline_id}")
        
        self.logger.info(f"Executing pipeline: {pipeline.name}")
        
        # Create execution context
        execution_id = f"exec_{pipeline_id}_{int(time.time())}"
        context = {
            'execution_id': execution_id,
            'pipeline_id': pipeline_id,
            'started_at': time.time(),
            'stage_results': {}
        }
        
        # Execute stages
        for stage in pipeline.stages:
            # Check dependencies
            if not self._check_dependencies(stage, context):
                self.logger.error(f"Dependencies not met for stage: {stage.name}")
                continue
            
            # Create task for stage
            stage_task = self._create_stage_task(stage, pipeline, context)
            
            # Route to appropriate node
            if stage.type == 'scrape':
                queue = self._get_scraper_queue(stage.config)
            elif stage.type == 'analyze':
                queue = 'ai_tasks:ai-node-01'
            elif stage.type == 'generate':
                queue = 'ai_tasks:ai-node-01'
            else:
                queue = f'tasks:{self.node_id}'  # Process locally
            
            # Queue task
            self.redis.lpush(queue, json.dumps(stage_task))
            
            # For sequential execution, wait for completion
            if stage.config.get('wait_for_completion', False):
                self._wait_for_stage_completion(execution_id, stage.id)
        
        # Update pipeline
        pipeline.last_run = time.time()
        pipeline.run_count = pipeline.run_count + 1 if hasattr(pipeline, 'run_count') else 1
        self._save_pipeline(pipeline)
        
        self.scheduler_metrics['pipelines_executed'] += 1
        
        return {
            'execution_id': execution_id,
            'status': 'executing',
            'stages_queued': len(pipeline.stages)
        }
    
    def schedule_task(self, config: Dict) -> Dict:
        """Schedule a recurring task"""
        task = ScheduledTask(
            id=f"sched_{int(time.time())}",
            pipeline_id=config['pipeline_id'],
            schedule_type=config['schedule_type'],
            schedule_data=config['schedule_data'],
            next_run=self._calculate_next_run(config)
        )
        
        self.scheduled_tasks[task.id] = task
        self._save_scheduled_task(task)
        
        self.scheduler_metrics['tasks_scheduled'] += 1
        
        return {
            'task_id': task.id,
            'next_run': datetime.fromtimestamp(task.next_run).isoformat()
        }
    
    def do_light_scrape(self, url: str, params: Dict) -> Dict:
        """Perform lightweight scraping when idle"""
        if not self.can_scrape:
            raise ValueError("Light scraping not available")
        
        import requests
        from bs4 import BeautifulSoup
        
        try:
            # Simple GET request
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ScrapeHive/1.0)'
            })
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract data based on params
            if params.get('extract') == 'links':
                data = [a.get('href') for a in soup.find_all('a', href=True)]
            elif params.get('extract') == 'images':
                data = [img.get('src') for img in soup.find_all('img', src=True)]
            elif params.get('extract') == 'text':
                data = soup.get_text(strip=True)[:1000]
            else:
                data = {
                    'title': soup.title.string if soup.title else '',
                    'links_count': len(soup.find_all('a')),
                    'images_count': len(soup.find_all('img'))
                }
            
            self.scheduler_metrics['light_scrapes'] += 1
            
            return {
                'url': url,
                'data': data,
                'status_code': response.status_code,
                'scraped_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Light scrape failed: {e}")
            raise
    
    def _check_scheduled_tasks(self):
        """Check and execute due scheduled tasks"""
        now = time.time()
        
        for task_id, task in self.scheduled_tasks.items():
            if task.active and task.next_run <= now:
                # Execute pipeline
                try:
                    self.execute_pipeline(task.pipeline_id)
                    
                    # Update task
                    task.last_run = now
                    task.next_run = self._calculate_next_run({
                        'schedule_type': task.schedule_type,
                        'schedule_data': task.schedule_data
                    })
                    
                    self._save_scheduled_task(task)
                    
                except Exception as e:
                    self.logger.error(f"Failed to execute scheduled task {task_id}: {e}")
    
    def _validate_pipeline(self, pipeline: Pipeline) -> bool:
        """Validate pipeline configuration"""
        if not pipeline.stages:
            return False
        
        # Check for circular dependencies
        for stage in pipeline.stages:
            if stage.id in stage.dependencies:
                return False
        
        return True
    
    def _calculate_next_run(self, config: Dict) -> float:
        """Calculate next run time based on schedule"""
        schedule_type = config['schedule_type']
        schedule_data = config['schedule_data']
        
        if schedule_type == 'once':
            # Run once at specified time
            return schedule_data['timestamp']
            
        elif schedule_type == 'interval':
            # Run at regular intervals
            interval = schedule_data['interval']  # seconds
            return time.time() + interval
            
        elif schedule_type == 'daily':
            # Run daily at specific time
            hour = schedule_data['hour']
            minute = schedule_data.get('minute', 0)
            
            next_run = datetime.now().replace(hour=hour, minute=minute, second=0)
            if next_run.timestamp() <= time.time():
                next_run += timedelta(days=1)
                
            return next_run.timestamp()
            
        elif schedule_type == 'cron':
            # Cron expression (simplified)
            # TODO: Implement full cron parser
            return time.time() + 3600  # Default to 1 hour
            
        else:
            return time.time() + 86400  # Default to 24 hours
    
    def _get_scraper_queue(self, stage_config: Dict) -> str:
        """Get appropriate scraper queue based on requirements"""
        # Find best scraper node
        required_capabilities = stage_config.get('capabilities', [])
        
        # Get all scraper nodes
        scraper_nodes = self.redis.keys('node:scraper-*:info')
        
        best_node = None
        best_score = 0
        
        for node_key in scraper_nodes:
            node_info = self.redis.hgetall(node_key)
            if not node_info:
                continue
                
            # Check capabilities
            capabilities = json.loads(node_info.get('capabilities', '[]'))
            score = len(set(required_capabilities) & set(capabilities))
            
            if score > best_score:
                best_score = score
                best_node = node_key.split(':')[1]
        
        if best_node:
            return f'scraping_tasks:{best_node}'
        else:
            # Default to first available scraper
            return 'scraping_tasks:scraper-01'
    
    def _create_stage_task(self, stage: PipelineStage, pipeline: Pipeline, context: Dict) -> Dict:
        """Create task for pipeline stage"""
        task = {
            'id': f"{context['execution_id']}_{stage.id}",
            'type': stage.config.get('task_type', stage.type),
            'target': stage.config.get('target', ''),
            'priority': stage.config.get('priority', 1),
            'status': 'pending',
            'created_at': time.time(),
            'metadata': {
                'pipeline_id': pipeline.id,
                'execution_id': context['execution_id'],
                'stage_id': stage.id,
                'stage_config': stage.config,
                'context': context
            }
        }
        
        # Add input from dependencies
        if stage.dependencies:
            inputs = {}
            for dep_id in stage.dependencies:
                if dep_id in context['stage_results']:
                    inputs[dep_id] = context['stage_results'][dep_id]
            task['metadata']['inputs'] = inputs
        
        return task
    
    def _save_pipeline(self, pipeline: Pipeline):
        """Save pipeline to Redis"""
        self.redis.hset(f'pipeline:{pipeline.id}', mapping=pipeline.to_dict())
    
    def _save_scheduled_task(self, task: ScheduledTask):
        """Save scheduled task"""
        self.redis.hset(f'scheduled:{task.id}', mapping=asdict(task))