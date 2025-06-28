# __init__.py (root)
"""
ScrapeHive - Distributed Content Intelligence System
"""

__version__ = "1.0.0"
__author__ = "ScrapeHive Team"
__description__ = "Distributed system for scraping, analyzing, and generating viral content"

from .config import Config
from .utils.logger import setup_logging

# nodes/__init__.py
"""
ScrapeHive Node Components
"""

from .base_node import BaseNode, NodeType, Task, TaskStatus
from .ai_node import AINode
from .scraping_node import ScrapingNode
from .scheduler_node import SchedulerNode
from .view_master import ViewMasterNode

__all__ = [
    'BaseNode', 'NodeType', 'Task', 'TaskStatus',
    'AINode', 'ScrapingNode', 'SchedulerNode', 'ViewMasterNode'
]

# scrapers/__init__.py
"""
Platform-specific scrapers
"""

from .reddit_scraper import RedditScraper
from .twitter_scraper import TwitterScraper
from .instagram_scraper import InstagramScraper
from .tiktok_scraper import TikTokScraper

__all__ = [
    'RedditScraper', 'TwitterScraper', 
    'InstagramScraper', 'TikTokScraper'
]

# models/__init__.py
"""
AI Models and Content Analysis
"""

from .viral_predictor import ViralPredictor
from .content_analyzer import ContentAnalyzer
from .script_generator import ScriptGenerator
from .video_generator import VideoGenerator

__all__ = [
    'ViralPredictor', 'ContentAnalyzer',
    'ScriptGenerator', 'VideoGenerator'
]

# utils/__init__.py
"""
Utility modules for ScrapeHive
"""

from .anti_detection import AntiDetection
from .rate_limiter import EnhancedRateLimiter
from .proxy_providers import ProxyProvider
from .pipeline_manager import PipelineManager
from .asset_manager import AssetManager

__all__ = [
    'AntiDetection', 'EnhancedRateLimiter', 'ProxyProvider',
    'PipelineManager', 'AssetManager'
]

# dashboard/__init__.py
"""
Web Dashboard Components
"""

from .view_master import ViewMasterNode

__all__ = ['ViewMasterNode']

# auth/__init__.py
"""
Authentication and Security
"""

from .session_manager import SessionManager
from .token_manager import TokenManager

__all__ = ['SessionManager', 'TokenManager']

# database/__init__.py
"""
Database models and utilities
"""

from .connection import DatabaseManager
from .models import *

__all__ = ['DatabaseManager']

# api/__init__.py
"""
REST API components
"""

from .app import create_app

__all__ = ['create_app']

# tests/__init__.py
"""
Test utilities
"""

# monitoring/__init__.py
"""
Monitoring and metrics
"""

from .metrics_collector import MetricsCollector
from .health_checker import HealthChecker

__all__ = ['MetricsCollector', 'HealthChecker']