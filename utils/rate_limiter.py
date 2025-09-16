# utils/rate_limiter.py
"""
Enhanced rate limiter with platform-specific rules
Implements token bucket algorithm with burst capacity
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class RateLimitRule:
    """Rate limit rule for a specific endpoint"""
    requests_per_minute: int
    requests_per_hour: int
    burst_capacity: int  # Allow burst requests
    cooldown_seconds: float
    
@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    last_update: float
    refill_rate: float  # tokens per second
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        now = time.time()
        
        # Refill tokens based on time passed
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def time_until_tokens(self, tokens: int = 1) -> float:
        """Time until we have enough tokens"""
        if self.tokens >= tokens:
            return 0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate

class EnhancedRateLimiter:
    """
    Advanced rate limiter with multiple strategies:
    - Token bucket for burst handling
    - Sliding window for accurate rate limiting
    - Platform-specific rules
    - Automatic backoff on 429s
    """
    
    def __init__(self, platform: str):
        self.platform = platform
        self.logger = logging.getLogger(f'rate_limiter.{platform}')
        
        # Platform-specific rules
        self.rules = self._get_platform_rules(platform)
        
        # Token buckets for different endpoints
        self.buckets = {}
        self._init_buckets()
        
        # Request history for sliding window
        self.request_history = defaultdict(list)
        
        # Backoff state
        self.backoff_until = 0
        self.consecutive_429s = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
    def _get_platform_rules(self, platform: str) -> Dict[str, RateLimitRule]:
        """Get platform-specific rate limit rules"""
        
        rules = {
            'reddit': {
                'api': RateLimitRule(
                    requests_per_minute=60,
                    requests_per_hour=600,
                    burst_capacity=10,
                    cooldown_seconds=1.0
                ),
                'json': RateLimitRule(
                    requests_per_minute=30,
                    requests_per_hour=300,
                    burst_capacity=5,
                    cooldown_seconds=2.0
                ),
                'web': RateLimitRule(
                    requests_per_minute=20,
                    requests_per_hour=200,
                    burst_capacity=3,
                    cooldown_seconds=3.0
                )
            },
            'twitter': {
                'api': RateLimitRule(
                    requests_per_minute=300,
                    requests_per_hour=3000,
                    burst_capacity=50,
                    cooldown_seconds=0.2
                ),
                'web': RateLimitRule(
                    requests_per_minute=30,
                    requests_per_hour=300,
                    burst_capacity=5,
                    cooldown_seconds=2.0
                )
            },
            'instagram': {
                'api': RateLimitRule(
                    requests_per_minute=20,
                    requests_per_hour=200,
                    burst_capacity=3,
                    cooldown_seconds=3.0
                ),
                'web': RateLimitRule(
                    requests_per_minute=15,
                    requests_per_hour=150,
                    burst_capacity=2,
                    cooldown_seconds=4.0
                )
            },
            'tiktok': {
                'api': RateLimitRule(
                    requests_per_minute=30,
                    requests_per_hour=300,
                    burst_capacity=5,
                    cooldown_seconds=2.0
                ),
                'web': RateLimitRule(
                    requests_per_minute=20,
                    requests_per_hour=200,
                    burst_capacity=3,
                    cooldown_seconds=3.0
                )
            }
        }
        
        # Default rules if platform not found
        default_rules = {
            'api': RateLimitRule(
                requests_per_minute=30,
                requests_per_hour=300,
                burst_capacity=5,
                cooldown_seconds=2.0
            ),
            'web': RateLimitRule(
                requests_per_minute=20,
                requests_per_hour=200,
                burst_capacity=3,
                cooldown_seconds=3.0
            )
        }
        
        return rules.get(platform, default_rules)
    
    def _init_buckets(self):
        """Initialize token buckets for each endpoint"""
        for endpoint, rule in self.rules.items():
            # Refill rate is requests per minute / 60
            refill_rate = rule.requests_per_minute / 60.0
            
            self.buckets[endpoint] = TokenBucket(
                capacity=rule.burst_capacity,
                tokens=rule.burst_capacity,  # Start full
                last_update=time.time(),
                refill_rate=refill_rate
            )
    
    def check_limit(self, endpoint: str = 'api', tokens: int = 1) -> bool:
        """Check if request is allowed"""
        with self.lock:
            # Check backoff
            if time.time() < self.backoff_until:
                return False
            
            # Get bucket
            bucket = self.buckets.get(endpoint)
            if not bucket:
                self.logger.warning(f"No bucket for endpoint: {endpoint}")
                return True
            
            # Check token bucket
            if not bucket.consume(tokens):
                return False
            
            # Check sliding window (hourly limit)
            rule = self.rules[endpoint]
            now = time.time()
            hour_ago = now - 3600
            
            # Clean old requests
            self.request_history[endpoint] = [
                t for t in self.request_history[endpoint] if t > hour_ago
            ]
            
            # Check hourly limit
            if len(self.request_history[endpoint]) >= rule.requests_per_hour:
                return False
            
            # Record request
            self.request_history[endpoint].append(now)
            
            # Reset consecutive 429s on successful check
            if self.consecutive_429s > 0:
                self.consecutive_429s = 0
            
            return True
    
    def get_wait_time(self, endpoint: str = 'api', tokens: int = 1) -> float:
        """Get time to wait before next request is allowed"""
        with self.lock:
            # Check backoff
            if time.time() < self.backoff_until:
                return self.backoff_until - time.time()
            
            # Get bucket
            bucket = self.buckets.get(endpoint)
            if not bucket:
                return 0
            
            # Get time until tokens available
            token_wait = bucket.time_until_tokens(tokens)
            
            # Check hourly limit
            rule = self.rules[endpoint]
            hour_ago = time.time() - 3600
            recent_requests = [
                t for t in self.request_history[endpoint] if t > hour_ago
            ]
            
            if len(recent_requests) >= rule.requests_per_hour:
                # Find when the oldest request will be outside the window
                oldest = min(recent_requests)
                hour_wait = (oldest + 3600) - time.time()
                return max(token_wait, hour_wait)
            
            return token_wait
    
    def handle_429(self, endpoint: str = 'api', retry_after: Optional[int] = None):
        """Handle 429 rate limit response"""
        with self.lock:
            self.consecutive_429s += 1
            
            # Calculate backoff
            if retry_after:
                backoff = retry_after
            else:
                # Exponential backoff: 60, 120, 240, 480...
                backoff = min(60 * (2 ** (self.consecutive_429s - 1)), 3600)
            
            self.backoff_until = time.time() + backoff
            
            self.logger.warning(
                f"Rate limit 429 received (#{self.consecutive_429s}). "
                f"Backing off for {backoff} seconds"
            )
            
            # Reduce tokens to prevent immediate retry
            bucket = self.buckets.get(endpoint)
            if bucket:
                bucket.tokens = 0
    
    def reset_endpoint(self, endpoint: str = 'api'):
        """Reset rate limit for endpoint (e.g., after IP rotation)"""
        with self.lock:
            # Reset bucket
            if endpoint in self.buckets:
                rule = self.rules[endpoint]
                self.buckets[endpoint] = TokenBucket(
                    capacity=rule.burst_capacity,
                    tokens=rule.burst_capacity / 2,  # Start half full
                    last_update=time.time(),
                    refill_rate=rule.requests_per_minute / 60.0
                )
            
            # Clear history
            self.request_history[endpoint].clear()
            
            # Reset backoff if all endpoints reset
            if not any(self.request_history.values()):
                self.backoff_until = 0
                self.consecutive_429s = 0
    
    def get_stats(self) -> Dict:
        """Get current rate limit statistics"""
        with self.lock:
            stats = {
                'platform': self.platform,
                'backoff_until': self.backoff_until,
                'consecutive_429s': self.consecutive_429s,
                'endpoints': {}
            }
            
            for endpoint, bucket in self.buckets.items():
                # Update tokens
                now = time.time()
                elapsed = now - bucket.last_update
                current_tokens = min(
                    bucket.capacity,
                    bucket.tokens + elapsed * bucket.refill_rate
                )
                
                # Count recent requests
                hour_ago = now - 3600
                minute_ago = now - 60
                
                hourly_requests = len([
                    t for t in self.request_history[endpoint] if t > hour_ago
                ])
                minute_requests = len([
                    t for t in self.request_history[endpoint] if t > minute_ago
                ])
                
                stats['endpoints'][endpoint] = {
                    'current_tokens': current_tokens,
                    'capacity': bucket.capacity,
                    'requests_last_minute': minute_requests,
                    'requests_last_hour': hourly_requests,
                    'limit_per_minute': self.rules[endpoint].requests_per_minute,
                    'limit_per_hour': self.rules[endpoint].requests_per_hour
                }
            
            return stats


class RateLimiter:
    """
    Backwards-compatible wrapper used by nodes.
    Defaults to 'web' endpoint checks for simplicity.
    """
    def __init__(self, platform: str = 'generic'):
        self._inner = EnhancedRateLimiter(platform)
    
    def can_scrape(self, platform: Optional[str] = None, endpoint: str = 'web') -> bool:
        return self._inner.check_limit(endpoint)
    
    def get_wait_time(self, platform: Optional[str] = None, endpoint: str = 'web') -> float:
        return self._inner.get_wait_time(endpoint)
    
    def handle_429(self, platform: Optional[str] = None, endpoint: str = 'web', retry_after: Optional[int] = None):
        return self._inner.handle_429(endpoint, retry_after)
    
    def get_current_limits(self):
        return self._inner.get_stats()