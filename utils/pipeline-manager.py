# utils/pipeline_manager.py
"""
Advanced Pipeline Management System
Handles complex multi-stage workflows with dependencies
"""

import uuid
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import networkx as nx
from datetime import datetime, timedelta
import logging

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"

class StageType(Enum):
    SCRAPE = "scrape"
    ANALYZE = "analyze"
    GENERATE = "generate"
    FILTER = "filter"
    TRANSFORM = "transform"
    STORE = "store"
    NOTIFY = "notify"
    CUSTOM = "custom"

@dataclass
class PipelineStage:
    """Represents a single stage in a pipeline"""
    id: str
    name: str
    type: StageType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Dict = field(default_factory=lambda: {"max_retries": 3, "backoff": "exponential"})
    timeout: int = 300  # 5 minutes default
    condition: Optional[str] = None  # Python expression for conditional execution
    
@dataclass
class PipelineExecution:
    """Tracks a pipeline execution instance"""
    id: str
    pipeline_id: str
    started_at: float
    status: str
    stage_results: Dict[str, Any] = field(default_factory=dict)
    stage_statuses: Dict[str, StageStatus] = field(default_factory=dict)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Pipeline:
    """Pipeline definition"""
    id: str
    name: str
    description: str
    stages: List[PipelineStage]
    schedule: Optional[Dict] = None
    active: bool = True
    created_at: float = field(default_factory=time.time)
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['stages'] = [asdict(stage) for stage in self.stages]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Pipeline':
        stages = [PipelineStage(**stage) for stage in data.pop('stages', [])]
        return cls(stages=stages, **data)

class PipelineManager:
    """
    Advanced pipeline management with:
    - DAG-based execution
    - Conditional stages
    - Parallel execution
    - Error handling
    - State persistence
    """
    
    def __init__(self, redis_client, node_id: str):
        self.redis = redis_client
        self.node_id = node_id
        self.logger = logging.getLogger(f'pipeline_manager.{node_id}')
        self.executions = {}
        self.stage_handlers = self._init_stage_handlers()
        
    def _init_stage_handlers(self) -> Dict[StageType, Callable]:
        """Initialize handlers for each stage type"""
        return {
            StageType.SCRAPE: self._handle_scrape_stage,
            StageType.ANALYZE: self._handle_analyze_stage,
            StageType.GENERATE: self._handle_generate_stage,
            StageType.FILTER: self._handle_filter_stage,
            StageType.TRANSFORM: self._handle_transform_stage,
            StageType.STORE: self._handle_store_stage,
            StageType.NOTIFY: self._handle_notify_stage,
            StageType.CUSTOM: self._handle_custom_stage
        }
    
    def create_pipeline(self, pipeline_config: Dict) -> Pipeline:
        """Create a new pipeline from configuration"""
        # Validate pipeline
        self._validate_pipeline_config(pipeline_config)
        
        # Create pipeline object
        pipeline = Pipeline(
            id=f"pipe_{uuid.uuid4().hex[:8]}",
            name=pipeline_config['name'],
            description=pipeline_config.get('description', ''),
            stages=[],
            schedule=pipeline_config.get('schedule'),
            tags=pipeline_config.get('tags', []),
            config=pipeline_config.get('config', {})
        )
        
        # Add stages
        for stage_config in pipeline_config['stages']:
            stage = PipelineStage(
                id=f"stage_{len(pipeline.stages)}",
                name=stage_config['name'],
                type=StageType(stage_config['type']),
                config=stage_config.get('config', {}),
                dependencies=stage_config.get('dependencies', []),
                retry_policy=stage_config.get('retry_policy', {"max_retries": 3}),
                timeout=stage_config.get('timeout', 300),
                condition=stage_config.get('condition')
            )
            pipeline.stages.append(stage)
        
        # Validate DAG (no cycles)
        if not self._validate_dag(pipeline):
            raise ValueError("Pipeline contains circular dependencies")
        
        # Save pipeline
        self._save_pipeline(pipeline)
        
        self.logger.info(f"Created pipeline: {pipeline.name} ({pipeline.id})")
        return pipeline
    
    def execute_pipeline(self, pipeline_id: str, context: Dict = None) -> str:
        """Execute a pipeline with given context"""
        # Load pipeline
        pipeline = self._load_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        if not pipeline.active:
            raise ValueError(f"Pipeline is not active: {pipeline_id}")
        
        # Create execution instance
        execution = PipelineExecution(
            id=f"exec_{uuid.uuid4().hex[:8]}",
            pipeline_id=pipeline_id,
            started_at=time.time(),
            status="running",
            context=context or {}
        )
        
        # Initialize stage statuses
        for stage in pipeline.stages:
            execution.stage_statuses[stage.id] = StageStatus.PENDING
        
        # Store execution
        self.executions[execution.id] = execution
        self._save_execution(execution)
        
        # Start execution in background
        asyncio.create_task(self._execute_pipeline_async(pipeline, execution))
        
        self.logger.info(f"Started pipeline execution: {execution.id}")
        return execution.id
    
    async def _execute_pipeline_async(self, pipeline: Pipeline, execution: PipelineExecution):
        """Execute pipeline asynchronously"""
        try:
            # Build execution graph
            graph = self._build_execution_graph(pipeline)
            
            # Execute stages in topological order
            for stage_id in nx.topological_sort(graph):
                stage = next(s for s in pipeline.stages if s.id == stage_id)
                
                # Check if dependencies completed
                if not self._check_dependencies(stage, execution):
                    execution.stage_statuses[stage.id] = StageStatus.SKIPPED
                    continue
                
                # Check condition
                if stage.condition and not self._evaluate_condition(stage.condition, execution):
                    execution.stage_statuses[stage.id] = StageStatus.SKIPPED
                    self.logger.info(f"Skipping stage {stage.name} due to condition")
                    continue
                
                # Execute stage
                await self._execute_stage(stage, execution)
            
            # Pipeline completed
            execution.status = "completed"
            execution.completed_at = time.time()
            
            # Update pipeline last run
            pipeline.last_run = execution.completed_at
            self._save_pipeline(pipeline)
            
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = time.time()
            self.logger.error(f"Pipeline execution failed: {e}")
        
        finally:
            self._save_execution(execution)
    
    async def _execute_stage(self, stage: PipelineStage, execution: PipelineExecution):
        """Execute a single stage with retry logic"""
        execution.stage_statuses[stage.id] = StageStatus.RUNNING
        self._save_execution(execution)
        
        retries = 0
        max_retries = stage.retry_policy.get('max_retries', 3)
        
        while retries <= max_retries:
            try:
                # Get handler for stage type
                handler = self.stage_handlers.get(stage.type)
                if not handler:
                    raise ValueError(f"No handler for stage type: {stage.type}")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(stage, execution),
                    timeout=stage.timeout
                )
                
                # Store result
                execution.stage_results[stage.id] = result
                execution.stage_statuses[stage.id] = StageStatus.COMPLETED
                self._save_execution(execution)
                
                self.logger.info(f"Stage completed: {stage.name}")
                return
                
            except asyncio.TimeoutError:
                self.logger.error(f"Stage {stage.name} timed out")
                retries += 1
                
            except Exception as e:
                self.logger.error(f"Stage {stage.name} failed: {e}")
                retries += 1
                
                if retries <= max_retries:
                    # Calculate backoff
                    backoff = self._calculate_backoff(retries, stage.retry_policy)
                    self.logger.info(f"Retrying stage {stage.name} in {backoff}s")
                    await asyncio.sleep(backoff)
        
        # Stage failed after all retries
        execution.stage_statuses[stage.id] = StageStatus.FAILED
        self._save_execution(execution)
        raise Exception(f"Stage {stage.name} failed after {max_retries} retries")
    
    async def _handle_scrape_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle scrape stage"""
        config = stage.config
        
        # Create scraping task
        task = {
            'id': f"{execution.id}_{stage.id}",
            'type': 'scrape_url',
            'target': config['url'],
            'priority': 2,
            'metadata': {
                'platform': config.get('platform', 'auto'),
                'params': config.get('params', {}),
                'execution_id': execution.id,
                'stage_id': stage.id
            }
        }
        
        # Queue to scraping node
        queue = self._get_scraper_queue(config.get('capabilities', []))
        self.redis.lpush(queue, json.dumps(task))
        
        # Wait for result
        result_key = f"stage_result:{execution.id}:{stage.id}"
        
        timeout = stage.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.redis.get(result_key)
            if result:
                return json.loads(result)
            await asyncio.sleep(1)
        
        raise asyncio.TimeoutError(f"Scraping stage timed out")
    
    async def _handle_analyze_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle analyze stage"""
        config = stage.config
        
        # Get input data
        input_data = self._get_stage_inputs(stage, execution)
        
        # Create analysis task
        task = {
            'id': f"{execution.id}_{stage.id}",
            'type': config.get('analysis_type', 'analyze_content'),
            'target': 'analysis',
            'priority': 1,
            'metadata': {
                'data': input_data,
                'model': config.get('model', 'content_analyzer'),
                'params': config.get('params', {}),
                'execution_id': execution.id,
                'stage_id': stage.id
            }
        }
        
        # Queue to AI node
        self.redis.lpush('ai_tasks:ai-node-01', json.dumps(task))
        
        # Wait for result
        result_key = f"stage_result:{execution.id}:{stage.id}"
        
        timeout = stage.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.redis.get(result_key)
            if result:
                return json.loads(result)
            await asyncio.sleep(1)
        
        raise asyncio.TimeoutError(f"Analysis stage timed out")
    
    async def _handle_filter_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle filter stage"""
        config = stage.config
        input_data = self._get_stage_inputs(stage, execution)
        
        # Apply filters
        filtered_data = []
        
        for item in input_data:
            # Check all filter conditions
            passed = True
            
            for filter_rule in config.get('filters', []):
                field = filter_rule['field']
                operator = filter_rule['operator']
                value = filter_rule['value']
                
                item_value = self._get_nested_value(item, field)
                
                if not self._evaluate_filter(item_value, operator, value):
                    passed = False
                    break
            
            if passed:
                filtered_data.append(item)
        
        return {
            'filtered_count': len(filtered_data),
            'total_count': len(input_data),
            'data': filtered_data
        }
    
    async def _handle_transform_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle transform stage"""
        config = stage.config
        input_data = self._get_stage_inputs(stage, execution)
        
        # Apply transformations
        transformed_data = []
        
        for item in input_data:
            transformed_item = {}
            
            # Apply field mappings
            for mapping in config.get('mappings', []):
                source = mapping['source']
                target = mapping['target']
                transform = mapping.get('transform')
                
                value = self._get_nested_value(item, source)
                
                if transform:
                    value = self._apply_transform(value, transform)
                
                self._set_nested_value(transformed_item, target, value)
            
            transformed_data.append(transformed_item)
        
        return {
            'count': len(transformed_data),
            'data': transformed_data
        }
    
    async def _handle_store_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle store stage"""
        config = stage.config
        input_data = self._get_stage_inputs(stage, execution)
        
        storage_type = config.get('storage_type', 'redis')
        
        if storage_type == 'redis':
            key = config.get('key', f"pipeline_data:{execution.id}")
            self.redis.set(key, json.dumps(input_data))
            self.redis.expire(key, config.get('ttl', 86400))  # 24 hours default
            
        elif storage_type == 'file':
            import os
            filename = config.get('filename', f"{execution.id}.json")
            filepath = os.path.join('data/pipeline_outputs', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(input_data, f, indent=2)
        
        elif storage_type == 'database':
            # Store in SQLite database
            # Implementation depends on database schema
            pass
        
        return {
            'storage_type': storage_type,
            'items_stored': len(input_data) if isinstance(input_data, list) else 1
        }
    
    async def _handle_notify_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle notification stage"""
        config = stage.config
        
        # Get notification data
        data = self._get_stage_inputs(stage, execution)
        
        notification_type = config.get('type', 'log')
        
        if notification_type == 'log':
            self.logger.info(f"Pipeline notification: {config.get('message', 'Pipeline stage completed')}")
            
        elif notification_type == 'webhook':
            webhook_url = config.get('webhook_url')
            if webhook_url:
                import requests
                requests.post(webhook_url, json={
                    'pipeline_id': execution.pipeline_id,
                    'execution_id': execution.id,
                    'stage': stage.name,
                    'data': data
                })
        
        elif notification_type == 'email':
            # Email notification implementation
            pass
        
        return {'notified': True, 'type': notification_type}
    
    async def _handle_custom_stage(self, stage: PipelineStage, execution: PipelineExecution) -> Dict:
        """Handle custom stage with user-defined logic"""
        config = stage.config
        
        # Execute custom Python code (sandboxed)
        code = config.get('code', '')
        inputs = self._get_stage_inputs(stage, execution)
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max
            }
        }
        
        safe_locals = {
            'inputs': inputs,
            'config': config,
            'context': execution.context
        }
        
        try:
            exec(code, safe_globals, safe_locals)
            return safe_locals.get('output', {})
        except Exception as e:
            raise Exception(f"Custom stage error: {e}")
    
    def _build_execution_graph(self, pipeline: Pipeline) -> nx.DiGraph:
        """Build directed graph of pipeline stages"""
        graph = nx.DiGraph()
        
        # Add all stages as nodes
        for stage in pipeline.stages:
            graph.add_node(stage.id)
        
        # Add dependencies as edges
        for stage in pipeline.stages:
            for dep_id in stage.dependencies:
                graph.add_edge(dep_id, stage.id)
        
        return graph
    
    def _validate_dag(self, pipeline: Pipeline) -> bool:
        """Validate that pipeline is a valid DAG (no cycles)"""
        graph = self._build_execution_graph(pipeline)
        return nx.is_directed_acyclic_graph(graph)
    
    def _check_dependencies(self, stage: PipelineStage, execution: PipelineExecution) -> bool:
        """Check if all dependencies are completed"""
        for dep_id in stage.dependencies:
            if execution.stage_statuses.get(dep_id) != StageStatus.COMPLETED:
                return False
        return True
    
    def _evaluate_condition(self, condition: str, execution: PipelineExecution) -> bool:
        """Evaluate stage condition"""
        try:
            # Create safe evaluation context
            context = {
                'results': execution.stage_results,
                'context': execution.context,
                'time': time.time()
            }
            
            # Evaluate condition
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return True  # Default to executing stage
    
    def _get_stage_inputs(self, stage: PipelineStage, execution: PipelineExecution) -> Any:
        """Get inputs for a stage from dependencies"""
        if not stage.dependencies:
            return execution.context.get('initial_data', {})
        
        if len(stage.dependencies) == 1:
            # Single dependency - return its output directly
            dep_result = execution.stage_results.get(stage.dependencies[0], {})
            return dep_result.get('data', dep_result)
        
        # Multiple dependencies - return as dict
        inputs = {}
        for dep_id in stage.dependencies:
            dep_result = execution.stage_results.get(dep_id, {})
            inputs[dep_id] = dep_result.get('data', dep_result)
        
        return inputs
    
    def _get_scraper_queue(self, capabilities: List[str]) -> str:
        """Get appropriate scraper queue based on capabilities"""
        # Simple logic - can be enhanced
        if 'browser' in capabilities:
            return 'scraping_tasks:scraper-laptop-01'
        else:
            return 'scraping_tasks:scraper-rpi-01'
    
    def _calculate_backoff(self, attempt: int, policy: Dict) -> float:
        """Calculate backoff time based on retry policy"""
        strategy = policy.get('backoff', 'exponential')
        
        if strategy == 'exponential':
            return min(2 ** attempt, 300)  # Max 5 minutes
        elif strategy == 'linear':
            return attempt * 10
        else:
            return 30  # Fixed 30 seconds
    
    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get nested value from dict using dot notation"""
        parts = path.split('.')
        value = obj
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        
        return value
    
    def _set_nested_value(self, obj: Dict, path: str, value: Any):
        """Set nested value in dict using dot notation"""
        parts = path.split('.')
        
        for i, part in enumerate(parts[:-1]):
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        
        obj[parts[-1]] = value
    
    def _evaluate_filter(self, value: Any, operator: str, target: Any) -> bool:
        """Evaluate filter condition"""
        if operator == 'equals':
            return value == target
        elif operator == 'not_equals':
            return value != target
        elif operator == 'greater_than':
            return value > target
        elif operator == 'less_than':
            return value < target
        elif operator == 'contains':
            return target in str(value)
        elif operator == 'regex':
            import re
            return bool(re.search(target, str(value)))
        else:
            return True
    
    def _apply_transform(self, value: Any, transform: str) -> Any:
        """Apply transformation to value"""
        if transform == 'uppercase':
            return str(value).upper()
        elif transform == 'lowercase':
            return str(value).lower()
        elif transform == 'trim':
            return str(value).strip()
        elif transform.startswith('slice:'):
            # slice:0:100 - get first 100 chars
            parts = transform.split(':')
            start = int(parts[1])
            end = int(parts[2])
            return str(value)[start:end]
        else:
            return value
    
    def _validate_pipeline_config(self, config: Dict):
        """Validate pipeline configuration"""
        if 'name' not in config:
            raise ValueError("Pipeline must have a name")
        
        if 'stages' not in config or not config['stages']:
            raise ValueError("Pipeline must have at least one stage")
        
        # Validate each stage
        for stage in config['stages']:
            if 'name' not in stage:
                raise ValueError("Each stage must have a name")
            if 'type' not in stage:
                raise ValueError("Each stage must have a type")
    
    def _save_pipeline(self, pipeline: Pipeline):
        """Save pipeline to Redis"""
        self.redis.hset(
            f"pipeline:{pipeline.id}",
            mapping=pipeline.to_dict()
        )
    
    def _load_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Load pipeline from Redis"""
        data = self.redis.hgetall(f"pipeline:{pipeline_id}")
        if data:
            return Pipeline.from_dict(data)
        return None
    
    def _save_execution(self, execution: PipelineExecution):
        """Save execution state"""
        self.redis.hset(
            f"execution:{execution.id}",
            mapping=asdict(execution)
        )
        
        # Also update pipeline executions list
        self.redis.lpush(
            f"pipeline_executions:{execution.pipeline_id}",
            execution.id
        )
        self.redis.ltrim(f"pipeline_executions:{execution.pipeline_id}", 0, 99)  # Keep last 100


# Example pipeline configurations
EXAMPLE_PIPELINES = {
    "reddit_to_video": {
        "name": "Reddit Story to Video Pipeline",
        "description": "Scrape Reddit stories, analyze for virality, generate scripts",
        "stages": [
            {
                "name": "Scrape Reddit",
                "type": "scrape",
                "config": {
                    "url": "https://reddit.com/r/AskReddit/top",
                    "platform": "reddit",
                    "params": {"limit": 50, "time_filter": "day"}
                }
            },
            {
                "name": "Filter High Engagement",
                "type": "filter",
                "dependencies": ["stage_0"],
                "config": {
                    "filters": [
                        {"field": "score", "operator": "greater_than", "value": 1000},
                        {"field": "num_comments", "operator": "greater_than", "value": 100}
                    ]
                }
            },
            {
                "name": "Analyze Virality",
                "type": "analyze",
                "dependencies": ["stage_1"],
                "config": {
                    "analysis_type": "predict_virality",
                    "model": "viral_predictor"
                }
            },
            {
                "name": "Generate Scripts",
                "type": "generate",
                "dependencies": ["stage_2"],
                "condition": "results['stage_2']['viral_score'] > 75",
                "config": {
                    "generation_type": "video_script",
                    "style": "engaging",
                    "duration": 60
                }
            },
            {
                "name": "Store Results",
                "type": "store",
                "dependencies": ["stage_3"],
                "config": {
                    "storage_type": "file",
                    "filename": "viral_scripts_{timestamp}.json"
                }
            }
        ],
        "schedule": {
            "type": "cron",
            "expression": "0 */6 * * *"  # Every 6 hours
        }
    },
    
    "multi_platform_trending": {
        "name": "Multi-Platform Trending Monitor",
        "description": "Monitor trending content across platforms",
        "stages": [
            {
                "name": "Scrape Reddit Trending",
                "type": "scrape",
                "config": {
                    "url": "https://reddit.com/r/all/rising",
                    "platform": "reddit",
                    "params": {"limit": 25}
                }
            },
            {
                "name": "Scrape Twitter Trending",
                "type": "scrape",
                "config": {
                    "url": "https://twitter.com/explore/tabs/trending",
                    "platform": "twitter",
                    "capabilities": ["browser"]
                }
            },
            {
                "name": "Combine Results",
                "type": "transform",
                "dependencies": ["stage_0", "stage_1"],
                "config": {
                    "mappings": [
                        {"source": "title", "target": "content"},
                        {"source": "score", "target": "engagement"}
                    ]
                }
            },
            {
                "name": "Detect Trends",
                "type": "analyze",
                "dependencies": ["stage_2"],
                "config": {
                    "analysis_type": "detect_trends",
                    "timeframe": "24h"
                }
            },
            {
                "name": "Notify on Hot Trends",
                "type": "notify",
                "dependencies": ["stage_3"],
                "condition": "len(results['stage_3']['emerging_topics']) > 0",
                "config": {
                    "type": "webhook",
                    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                    "message": "New trending topics detected!"
                }
            }
        ]
    }
}