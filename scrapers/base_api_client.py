# scrapers/base_api_client.py
"""
Base API client for compliant platform scraping
Provides common functionality for all platform APIs
"""

import time
import json
import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os
from urllib.parse import urlencode, urljoin
import yaml

class ComplianceMode(Enum):
    API_ONLY = "api_only"
    API_FIRST = "api_first"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class RateLimitStrategy(Enum):
    STRICT = "strict"
    ADAPTIVE = "adaptive"
    BURST = "burst"

@dataclass
class APIResponse:
    success: bool
    data: Any
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[int] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    method_used: str = "api"

class BaseAPIClient(ABC):
    """Base class for all platform API clients"""

    def __init__(self, platform: str, config: Dict):
        self.platform = platform
        self.config = config
        self.logger = logging.getLogger(f'api_client.{platform}')

        # Load compliance mode
        self.compliance_mode = ComplianceMode(
            os.getenv('COMPLIANCE_MODE', 'api_first')
        )

        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self._get_user_agent(),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })

        # Rate limiting
        self.rate_limiter = TokenBucketRateLimiter(
            platform, config.get('rate_limits', {})
        )

        # Stats tracking
        self.stats = {
            'api_calls': 0,
            'api_success': 0,
            'api_errors': 0,
            'fallback_calls': 0,
            'rate_limits_hit': 0,
            'last_reset': time.time()
        }

    def _get_user_agent(self) -> str:
        """Get appropriate user agent based on compliance mode"""
        if self.compliance_mode == ComplianceMode.API_ONLY:
            # Honest identification for API-only mode
            return f"ScrapeHive/1.0 (+https://scrapehive.com/bot) {self.platform}-client"
        else:
            # Standard browser user agent for fallback methods
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def search(self, query: str, limit: int = 25, **kwargs) -> APIResponse:
        """Main search method with compliance-aware fallback"""
        self.logger.info(f"Searching {self.platform} for: {query[:50]}...")

        # Try API first (if enabled and compliant mode allows)
        if self._should_use_api():
            try:
                return self._api_search(query, limit, **kwargs)
            except RateLimitExceededException as e:
                self.logger.warning(f"API rate limit exceeded: {e}")
                self.stats['rate_limits_hit'] += 1
                if self.compliance_mode == ComplianceMode.API_ONLY:
                    return APIResponse(
                        success=False,
                        data=None,
                        error_message="API rate limit exceeded and fallback disabled"
                    )
            except APIException as e:
                self.logger.error(f"API error: {e}")
                self.stats['api_errors'] += 1

        # Fallback methods (if compliance mode allows)
        if self._should_use_fallback():
            return self._fallback_search(query, limit, **kwargs)

        return APIResponse(
            success=False,
            data=None,
            error_message="No available methods for this request"
        )

    def _should_use_api(self) -> bool:
        """Check if API should be used based on compliance mode and limits"""
        api_config = self.config.get('api', {})

        if not api_config.get('enabled', False):
            return False

        if not self._has_api_credentials():
            return False

        if not self.rate_limiter.can_make_request('api'):
            return False

        return True

    def _should_use_fallback(self) -> bool:
        """Check if fallback methods should be used"""
        if self.compliance_mode == ComplianceMode.API_ONLY:
            return False

        fallback_config = self.config.get('fallback', {})
        return fallback_config.get('enabled', False)

    @abstractmethod
    def _has_api_credentials(self) -> bool:
        """Check if platform has required API credentials"""
        pass

    @abstractmethod
    def _api_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Implement platform-specific API search"""
        pass

    def _fallback_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Fallback search using compliant scraping methods"""
        self.logger.info(f"Using fallback methods for {self.platform}")

        fallback_config = self.config.get('fallback', {})
        methods = fallback_config.get('methods', [])

        for method in methods:
            try:
                if method == "json_endpoints":
                    return self._json_endpoint_search(query, limit, **kwargs)
                elif method == "public_api":
                    return self._public_api_search(query, limit, **kwargs)
                elif method == "rss_feeds":
                    return self._rss_search(query, limit, **kwargs)
                elif method == "embedded_data":
                    return self._embedded_data_search(query, limit, **kwargs)

            except Exception as e:
                self.logger.warning(f"Fallback method {method} failed: {e}")
                continue

        return APIResponse(
            success=False,
            data=None,
            error_message="All fallback methods failed"
        )

    def _json_endpoint_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search using public JSON endpoints (e.g., Reddit's .json URLs)"""
        # Implementation varies by platform
        return APIResponse(
            success=False,
            data=None,
            error_message="JSON endpoint search not implemented"
        )

    def _public_api_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search using public APIs that don't require authentication"""
        return APIResponse(
            success=False,
            data=None,
            error_message="Public API search not implemented"
        )

    def _rss_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search using RSS feeds"""
        return APIResponse(
            success=False,
            data=None,
            error_message="RSS search not implemented"
        )

    def _embedded_data_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Extract data from embedded JSON in web pages"""
        return APIResponse(
            success=False,
            data=None,
            error_message="Embedded data search not implemented"
        )

    def _make_api_request(self, endpoint: str, params: Dict = None) -> requests.Response:
        """Make API request with proper rate limiting and error handling"""

        # Check rate limit
        if not self.rate_limiter.can_make_request('api'):
            wait_time = self.rate_limiter.get_wait_time('api')
            raise RateLimitExceededException(f"Rate limit exceeded, wait {wait_time} seconds")

        # Add authentication
        headers = self._get_auth_headers()

        # Make request
        try:
            response = self.session.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=self.config.get('timeout', 30)
            )

            self.stats['api_calls'] += 1

            # Update rate limiter
            self.rate_limiter.record_request('api')

            # Check for rate limit headers
            self._update_rate_limit_info(response)

            if response.status_code == 429:
                raise RateLimitExceededException("HTTP 429 Too Many Requests")
            elif response.status_code >= 400:
                raise APIException(f"HTTP {response.status_code}: {response.text}")

            self.stats['api_success'] += 1
            return response

        except requests.exceptions.RequestException as e:
            self.stats['api_errors'] += 1
            raise APIException(f"Request failed: {e}")

    def _update_rate_limit_info(self, response: requests.Response):
        """Update rate limit information from response headers"""
        # Common header names across platforms
        header_mappings = {
            'x-ratelimit-remaining': 'remaining',
            'x-rate-limit-remaining': 'remaining',
            'x-ratelimit-reset': 'reset',
            'x-rate-limit-reset': 'reset',
        }

        rate_limit_info = {}
        for header, key in header_mappings.items():
            if header in response.headers:
                try:
                    rate_limit_info[key] = int(response.headers[header])
                except ValueError:
                    pass

        if rate_limit_info:
            self.rate_limiter.update_from_headers(rate_limit_info)

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for the platform"""
        pass

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            **self.stats,
            'rate_limiter_status': self.rate_limiter.get_status(),
            'compliance_mode': self.compliance_mode.value
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {key: 0 for key in self.stats}
        self.stats['last_reset'] = time.time()


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API requests"""

    def __init__(self, platform: str, config: Dict):
        self.platform = platform
        self.config = config
        self.tokens = {}
        self.last_refill = {}

        # Initialize buckets for different endpoint types
        for endpoint_type, limit in config.items():
            if isinstance(limit, int):
                self.tokens[endpoint_type] = limit
                self.last_refill[endpoint_type] = time.time()

    def can_make_request(self, endpoint_type: str) -> bool:
        """Check if request can be made"""
        if endpoint_type not in self.tokens:
            return True  # No limit configured

        self._refill_bucket(endpoint_type)
        return self.tokens[endpoint_type] > 0

    def record_request(self, endpoint_type: str):
        """Record that a request was made"""
        if endpoint_type in self.tokens:
            self.tokens[endpoint_type] = max(0, self.tokens[endpoint_type] - 1)

    def get_wait_time(self, endpoint_type: str) -> float:
        """Get time to wait before next request"""
        if endpoint_type not in self.config:
            return 0

        limit = self.config[endpoint_type]
        # Simple calculation - in practice would be more sophisticated
        return 60.0 / limit  # Assume per-minute limits

    def _refill_bucket(self, endpoint_type: str):
        """Refill token bucket based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill[endpoint_type]

        if elapsed > 60:  # Refill every minute
            limit = self.config.get(endpoint_type, 0)
            self.tokens[endpoint_type] = limit
            self.last_refill[endpoint_type] = now

    def update_from_headers(self, info: Dict):
        """Update rate limit info from API response headers"""
        # Could be used to sync with actual API limits
        pass

    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        return {
            'tokens': dict(self.tokens),
            'last_refill': dict(self.last_refill)
        }


class APIException(Exception):
    """Base exception for API errors"""
    pass


class RateLimitExceededException(APIException):
    """Exception raised when rate limit is exceeded"""
    pass


def load_platform_config(platform: str) -> Dict:
    """Load configuration for a specific platform"""
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'config', 'api_config.yaml'
    )

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Expand environment variables
        platform_config = config['platforms'].get(platform, {})
        return _expand_env_vars(platform_config)

    except Exception as e:
        logging.warning(f"Failed to load config for {platform}: {e}")
        return {}


def _expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in config"""
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        env_var = config[2:-1]
        default = None
        if ':-' in env_var:
            env_var, default = env_var.split(':-', 1)
        return os.getenv(env_var, default)
    else:
        return config