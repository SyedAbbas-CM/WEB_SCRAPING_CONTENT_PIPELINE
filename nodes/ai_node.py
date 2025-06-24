# nodes/ai_node.py
"""
AI Processing Node - Runs on GPU machine
Handles content analysis, viral prediction, script generation
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base_node import BaseNode, NodeType, Task
from models.viral_predictor import ViralPredictor
from models.content_analyzer import ContentAnalyzer
from models.script_generator import ScriptGenerator

@dataclass
class AITask:
    """Extended task for AI processing"""
    content: Dict
    model_type: str
    parameters: Dict

class AINode(BaseNode):
    """
    AI Processing Node
    - Viral content prediction
    - Content analysis and scoring
    - Script generation for videos
    - Trend detection
    """
    
    def __init__(self, node_id: str, master_ip: str = None):
        super().__init__(node_id, NodeType.AI_NODE, master_ip)
        
        # GPU setup
        self.device = self._setup_gpu()
        
        # Initialize models
        self.models = self._load_models()
        
        # Model performance tracking
        self.model_metrics = {
            'predictions_made': 0,
            'avg_inference_time': 0,
            'gpu_memory_used': 0
        }
        
    def _setup_gpu(self) -> str:
        """Setup GPU/CUDA"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available: {gpu_count} GPU(s)")
            
            # Log GPU details
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(
                    f"GPU {i}: {props.name} "
                    f"({props.total_memory / 1e9:.1f} GB, "
                    f"Compute {props.major}.{props.minor})"
                )
            
            # Set default GPU
            torch.cuda.set_device(0)
            return 'cuda:0'
        else:
            self.logger.warning("No CUDA available, using CPU")
            return 'cpu'
    
    def _load_models(self) -> Dict:
        """Load AI models"""
        models = {}
        model_path = os.getenv('MODEL_PATH', 'models/weights')
        
        try:
            # Viral prediction model
            self.logger.info("Loading viral prediction model...")
            models['viral_predictor'] = ViralPredictor(
                model_path=os.path.join(model_path, 'viral_predictor'),
                device=self.device
            )
            
            # Content analyzer
            self.logger.info("Loading content analyzer...")
            models['content_analyzer'] = ContentAnalyzer(
                model_path=os.path.join(model_path, 'content_analyzer'),
                device=self.device
            )
            
            # Script generator
            self.logger.info("Loading script generator...")
            models['script_generator'] = ScriptGenerator(
                model_path=os.path.join(model_path, 'script_generator'),
                device=self.device
            )
            
            self.logger.info(f"Loaded {len(models)} models successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            # Continue with mock models for testing
            models['viral_predictor'] = MockViralPredictor()
            models['content_analyzer'] = MockContentAnalyzer()
            models['script_generator'] = MockScriptGenerator()
            
        return models
    
    def get_node_info(self) -> Dict:
        """Override to add GPU info"""
        info = super().get_node_info()
        
        if self.device.startswith('cuda'):
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_used'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['gpu_utilization'] = self._get_gpu_utilization()
        
        info['model_metrics'] = self.model_metrics
        info['models_loaded'] = list(self.models.keys())
        
        return info
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
    
    def process_task(self, task: Task) -> Dict:
        """Process AI task"""
        task_type = task.type
        data = task.metadata or {}
        
        # Route to appropriate handler
        if task_type == 'predict_virality':
            return self.predict_virality(data)
        elif task_type == 'analyze_content':
            return self.analyze_content(data)
        elif task_type == 'generate_script':
            return self.generate_script(data)
        elif task_type == 'detect_trends':
            return self.detect_trends(data)
        else:
            raise ValueError(f"Unknown AI task type: {task_type}")
    
    def predict_virality(self, data: Dict) -> Dict:
        """Predict viral potential of content"""
        start_time = time.time()
        
        # Extract content
        content = data.get('content', {})
        platform = data.get('platform', 'unknown')
        
        # Get model
        predictor = self.models['viral_predictor']
        
        # Make prediction
        with torch.no_grad():
            prediction = predictor.predict(content, platform)
        
        # Calculate factors
        factors = self._calculate_viral_factors(content, platform)
        
        # Update metrics
        inference_time = time.time() - start_time
        self.model_metrics['predictions_made'] += 1
        self.model_metrics['avg_inference_time'] = (
            (self.model_metrics['avg_inference_time'] * (self.model_metrics['predictions_made'] - 1) + 
             inference_time) / self.model_metrics['predictions_made']
        )
        
        result = {
            'viral_score': prediction['score'],
            'confidence': prediction['confidence'],
            'factors': factors,
            'recommendation': 'CREATE' if prediction['score'] > 75 else 'SKIP',
            'optimal_posting_time': self._calculate_optimal_time(platform),
            'suggested_modifications': self._suggest_improvements(content, prediction),
            'inference_time': inference_time
        }
        
        self.logger.info(
            f"Viral prediction: score={prediction['score']:.1f}, "
            f"confidence={prediction['confidence']:.2f}, "
            f"recommendation={result['recommendation']}"
        )
        
        return result
    
    def analyze_content(self, data: Dict) -> Dict:
        """Deep content analysis"""
        content = data.get('content', {})
        
        # Get analyzer
        analyzer = self.models['content_analyzer']
        
        # Perform analysis
        with torch.no_grad():
            analysis = analyzer.analyze(content)
        
        # Extract insights
        result = {
            'content_id': content.get('id'),
            'sentiment': analysis['sentiment'],
            'emotions': analysis['emotions'],
            'topics': analysis['topics'],
            'keywords': analysis['keywords'],
            'readability_score': analysis['readability'],
            'engagement_prediction': analysis['engagement_score'],
            'content_quality': self._assess_quality(analysis),
            'target_audience': self._identify_audience(analysis),
            'recommendations': self._generate_recommendations(analysis)
        }
        
        return result
    
    def generate_script(self, data: Dict) -> Dict:
        """Generate video script from content"""
        content = data.get('content', {})
        style = data.get('style', 'engaging')
        duration = data.get('duration', 60)  # seconds
        
        # Get generator
        generator = self.models['script_generator']
        
        # Generate script
        with torch.no_grad():
            script = generator.generate(
                content=content,
                style=style,
                target_duration=duration
            )
        
        # Post-process script
        result = {
            'script_id': f"script_{int(time.time())}",
            'sections': script['sections'],
            'total_duration': script['duration'],
            'word_count': script['word_count'],
            'hooks': script['hooks'],
            'call_to_action': script['cta'],
            'voice_notes': self._generate_voice_notes(script),
            'visual_suggestions': self._suggest_visuals(script),
            'music_suggestions': self._suggest_music(style)
        }
        
        return result
    
    def detect_trends(self, data: Dict) -> Dict:
        """Detect trending topics and patterns"""
        contents = data.get('contents', [])
        timeframe = data.get('timeframe', '24h')
        
        # Analyze trends
        trends = {
            'emerging_topics': [],
            'declining_topics': [],
            'stable_trends': [],
            'viral_patterns': [],
            'recommendations': []
        }
        
        # Group by topics
        topic_groups = self._group_by_topics(contents)
        
        # Analyze each topic group
        for topic, items in topic_groups.items():
            trend_data = self._analyze_trend(items, timeframe)
            
            if trend_data['growth_rate'] > 50:
                trends['emerging_topics'].append({
                    'topic': topic,
                    'growth_rate': trend_data['growth_rate'],
                    'volume': trend_data['volume'],
                    'peak_time': trend_data['peak_time']
                })
            elif trend_data['growth_rate'] < -30:
                trends['declining_topics'].append({
                    'topic': topic,
                    'decline_rate': abs(trend_data['growth_rate']),
                    'peak_was': trend_data['peak_time']
                })
            else:
                trends['stable_trends'].append({
                    'topic': topic,
                    'consistency': trend_data['consistency']
                })
        
        # Identify viral patterns
        trends['viral_patterns'] = self._identify_viral_patterns(contents)
        
        # Generate recommendations
        trends['recommendations'] = self._generate_trend_recommendations(trends)
        
        return trends
    
    def _calculate_viral_factors(self, content: Dict, platform: str) -> Dict:
        """Calculate detailed viral factors"""
        factors = {
            'content_quality': 0,
            'timing_score': 0,
            'trend_alignment': 0,
            'emotional_impact': 0,
            'shareability': 0
        }
        
        # Content quality
        text = content.get('text', '') or content.get('title', '')
        factors['content_quality'] = min(100, len(text) / 2)  # Simple metric
        
        # Timing score
        hour = time.localtime().tm_hour
        optimal_hours = {
            'reddit': [9, 12, 17, 20],
            'tiktok': [6, 10, 19, 22],
            'instagram': [11, 14, 17, 20]
        }
        if hour in optimal_hours.get(platform, []):
            factors['timing_score'] = 90
        else:
            factors['timing_score'] = 50
        
        # Emotional impact
        emotional_words = ['amazing', 'unbelievable', 'shocking', 'heartwarming', 'incredible']
        factors['emotional_impact'] = sum(10 for word in emotional_words if word in text.lower())
        
        return factors
    
    def _calculate_optimal_time(self, platform: str) -> Dict:
        """Calculate optimal posting time"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        optimal_times = {
            'reddit': [9, 12, 17, 20],
            'tiktok': [6, 10, 19, 22],
            'instagram': [11, 14, 17, 20],
            'youtube': [14, 17, 20]
        }
        
        platform_times = optimal_times.get(platform, [12, 18])
        current_hour = now.hour
        
        # Find next optimal time
        for hour in platform_times:
            if hour > current_hour:
                optimal = now.replace(hour=hour, minute=0, second=0)
                break
        else:
            # Next day
            optimal = (now + timedelta(days=1)).replace(
                hour=platform_times[0], minute=0, second=0
            )
        
        return {
            'timestamp': optimal.isoformat(),
            'hours_from_now': (optimal - now).total_seconds() / 3600,
            'day_of_week': optimal.strftime('%A'),
            'is_weekend': optimal.weekday() >= 5
        }
    
    def _suggest_improvements(self, content: Dict, prediction: Dict) -> List[str]:
        """Suggest content improvements"""
        suggestions = []
        
        if prediction['score'] < 50:
            suggestions.append("Add more emotional triggers or compelling hooks")
            suggestions.append("Consider trending topics or current events")
        
        text = content.get('text', '')
        if len(text) < 100:
            suggestions.append("Expand content with more detail and context")
        elif len(text) > 500:
            suggestions.append("Consider breaking into multiple parts")
        
        if not any(char in text for char in ['?', '!']):
            suggestions.append("Add questions or exclamations for engagement")
        
        return suggestions
    
    def _assess_quality(self, analysis: Dict) -> Dict:
        """Assess content quality"""
        return {
            'overall_score': analysis.get('quality_score', 0),
            'grammar_score': analysis.get('grammar', 0),
            'clarity_score': analysis.get('clarity', 0),
            'originality_score': analysis.get('originality', 0),
            'depth_score': analysis.get('depth', 0)
        }
    
    def _generate_voice_notes(self, script: Dict) -> List[Dict]:
        """Generate voice direction notes"""
        notes = []
        
        for section in script['sections']:
            if section['type'] == 'hook':
                notes.append({
                    'section': section['id'],
                    'tone': 'excited',
                    'pace': 'quick',
                    'emphasis': section.get('emphasis_words', [])
                })
            elif section['type'] == 'story':
                notes.append({
                    'section': section['id'],
                    'tone': 'conversational',
                    'pace': 'moderate',
                    'pauses': section.get('pause_points', [])
                })
        
        return notes
    
    def _suggest_visuals(self, script: Dict) -> List[Dict]:
        """Suggest visuals for script"""
        visuals = []
        
        for section in script['sections']:
            if 'keywords' in section:
                visuals.append({
                    'section': section['id'],
                    'type': 'text_overlay',
                    'content': section['keywords'][:3],
                    'style': 'bold_centered'
                })
            
            if section['type'] == 'story':
                visuals.append({
                    'section': section['id'],
                    'type': 'background_video',
                    'suggestion': 'minecraft_parkour',
                    'alternative': 'subway_surfers'
                })
        
        return visuals

# Mock implementations for testing without real models
class MockViralPredictor:
    def predict(self, content, platform):
        # Simple mock scoring
        score = len(content.get('text', '')) % 100
        return {
            'score': score,
            'confidence': 0.75
        }

class MockContentAnalyzer:
    def analyze(self, content):
        return {
            'sentiment': 'positive',
            'emotions': ['joy', 'surprise'],
            'topics': ['technology', 'news'],
            'keywords': ['ai', 'automation', 'future'],
            'readability': 85,
            'engagement_score': 72,
            'quality_score': 80
        }

class MockScriptGenerator:
    def generate(self, content, style, target_duration):
        return {
            'sections': [
                {
                    'id': 'hook',
                    'type': 'hook',
                    'text': 'You won\'t believe what happened...',
                    'duration': 5
                },
                {
                    'id': 'main',
                    'type': 'story',
                    'text': content.get('text', '')[:200],
                    'duration': target_duration - 10
                },
                {
                    'id': 'cta',
                    'type': 'call_to_action',
                    'text': 'Follow for more!',
                    'duration': 5
                }
            ],
            'duration': target_duration,
            'word_count': 150,
            'hooks': ['You won\'t believe'],
            'cta': 'Follow for more!'
        }