# config/__init__.py
"""
Configuration management for ScrapeHive
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    decode_responses: bool = True

@dataclass
class DatabaseConfig:
    url: str = "sqlite:///scrapehive.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class ScrapingConfig:
    rate_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "reddit": {"requests_per_minute": 30, "cooldown_seconds": 2},
        "twitter": {"requests_per_minute": 20, "cooldown_seconds": 3},
        "instagram": {"requests_per_minute": 10, "cooldown_seconds": 6},
        "tiktok": {"requests_per_minute": 15, "cooldown_seconds": 4}
    })
    delays: Dict[str, float] = field(default_factory=lambda: {
        "min_delay": 2, "max_delay": 5, "human_variance": 0.3
    })
    user_agents: list = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ])

@dataclass
class AIConfig:
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    temperature: float = 0.7
    models_path: str = "models/weights"

@dataclass
class SecurityConfig:
    secret_key: str = "change-this-in-production"
    jwt_expiry_hours: int = 24
    api_key_required: bool = True
    allowed_ips: list = field(default_factory=list)
    rate_limit_per_ip: int = 1000

@dataclass
class Config:
    """Main configuration class"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Paths
    data_path: str = "data"
    logs_path: str = "logs"
    assets_path: str = "assets"
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, environment: Optional[str] = None) -> 'Config':
        """Load configuration from files and environment"""
        # Determine environment
        env = environment or os.getenv('SCRAPEHIVE_ENV', 'development')
        
        # Load base config
        config_dir = Path(config_path or 'config')
        base_config = {}
        
        # Load default config
        default_file = config_dir / 'default.yaml'
        if default_file.exists():
            with open(default_file, 'r') as f:
                base_config = yaml.safe_load(f) or {}
        
        # Load environment-specific config
        env_file = config_dir / f'{env}.yaml'
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                base_config = cls._deep_merge(base_config, env_config)
        
        # Override with environment variables
        base_config = cls._apply_env_overrides(base_config)
        
        # Create config object
        config = cls._dict_to_config(base_config)
        config.environment = env
        
        return config
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _apply_env_overrides(config: Dict) -> Dict:
        """Apply environment variable overrides"""
        env_mappings = {
            'REDIS_HOST': ['redis', 'host'],
            'REDIS_PORT': ['redis', 'port'],
            'REDIS_PASSWORD': ['redis', 'password'],
            'DATABASE_URL': ['database', 'url'],
            'LOG_LEVEL': ['log_level'],
            'SECRET_KEY': ['security', 'secret_key'],
            'AI_DEVICE': ['ai', 'device'],
            'MODELS_PATH': ['ai', 'models_path'],
            'DATA_PATH': ['data_path'],
            'ASSETS_PATH': ['assets_path']
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert types
                if env_var.endswith('_PORT'):
                    value = int(value)
                elif env_var.endswith('_DEBUG'):
                    value = value.lower() in ('true', '1', 'yes')
                
                # Set value in config
                current = config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = value
        
        return config
    
    @classmethod
    def _dict_to_config(cls, data: Dict) -> 'Config':
        """Convert dictionary to Config object"""
        # Extract nested configs
        redis_config = RedisConfig(**data.get('redis', {}))
        database_config = DatabaseConfig(**data.get('database', {}))
        scraping_config = ScrapingConfig(**data.get('scraping', {}))
        ai_config = AIConfig(**data.get('ai', {}))
        security_config = SecurityConfig(**data.get('security', {}))
        
        # Create main config
        main_data = {k: v for k, v in data.items() 
                    if k not in ['redis', 'database', 'scraping', 'ai', 'security']}
        
        return cls(
            redis=redis_config,
            database=database_config,
            scraping=scraping_config,
            ai=ai_config,
            security=security_config,
            **main_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'redis': self.redis.__dict__,
            'database': self.database.__dict__,
            'scraping': self.scraping.__dict__,
            'ai': self.ai.__dict__,
            'security': self.security.__dict__,
            'data_path': self.data_path,
            'logs_path': self.logs_path,
            'assets_path': self.assets_path
        }

# Global config instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config

def reload_config(config_path: Optional[str] = None, environment: Optional[str] = None):
    """Reload configuration"""
    global _config
    _config = Config.load(config_path, environment)