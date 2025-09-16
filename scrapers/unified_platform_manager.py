# scrapers/unified_platform_manager.py
"""
Unified Platform Manager
Coordinates all platform scrapers with compliance-first approach
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests

from .reddit_api_client import RedditAPIClient
from .youtube_api_client import YouTubeAPIClient
from .base_api_client import APIResponse, ComplianceMode
from ..utils.advanced_stealth_scraper import AdvancedStealthScraper


class Platform(Enum):
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"


@dataclass
class SearchRequest:
    platform: Platform
    query: str
    limit: int = 25
    filters: Dict[str, Any] = None
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


@dataclass
class AggregatedResult:
    success: bool
    results: Dict[str, Any]
    total_items: int
    platforms_used: List[str]
    methods_used: Dict[str, str]  # platform -> method
    errors: Dict[str, str]
    compliance_notes: List[str]


class UnifiedPlatformManager:
    """
    Unified manager for all platform scrapers with compliance-first approach
    """

    def __init__(self):
        self.logger = logging.getLogger('unified_platform_manager')

        # Load compliance mode
        self.compliance_mode = ComplianceMode(
            os.getenv('COMPLIANCE_MODE', 'api_first')
        )

        # Initialize platform clients
        self.clients = {}
        self._initialize_clients()

        # Request tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'api_requests': 0,
            'fallback_requests': 0,
            'rate_limited_requests': 0
        }

        # Rate limiting across platforms
        self.global_rate_limiter = GlobalRateLimiter()

        # Stealth scrapers (only initialized when needed)
        self.stealth_scrapers = {}

    def _initialize_clients(self):
        """Initialize platform API clients"""
        try:
            self.clients[Platform.REDDIT] = RedditAPIClient()
            self.logger.info("Reddit client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Reddit client: {e}")

        try:
            self.clients[Platform.YOUTUBE] = YouTubeAPIClient()
            self.logger.info("YouTube client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize YouTube client: {e}")

        # Add other platform clients as they're implemented
        # self.clients[Platform.TWITTER] = TwitterAPIClient()
        # self.clients[Platform.INSTAGRAM] = InstagramAPIClient()

    def search_single_platform(self, platform: Platform, query: str,
                             limit: int = 25, **kwargs) -> APIResponse:
        """Search a single platform with compliance controls"""

        if platform not in self.clients:
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Platform {platform.value} not supported"
            )

        self.request_stats['total_requests'] += 1

        # Check global rate limits
        if not self.global_rate_limiter.can_make_request(platform.value):
            wait_time = self.global_rate_limiter.get_wait_time(platform.value)
            self.logger.warning(f"Global rate limit hit for {platform.value}, waiting {wait_time}s")

            if self.compliance_mode in [ComplianceMode.API_ONLY, ComplianceMode.API_FIRST]:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"Rate limited for {platform.value}"
                )

        try:
            client = self.clients[platform]
            result = client.search(query, limit, **kwargs)

            # Update stats
            if result.success:
                self.request_stats['successful_requests'] += 1
                if result.method_used and 'api' in result.method_used:
                    self.request_stats['api_requests'] += 1
                else:
                    self.request_stats['fallback_requests'] += 1
            else:
                self.request_stats['failed_requests'] += 1

            return result

        except Exception as e:
            self.logger.error(f"Search failed for {platform.value}: {e}")
            self.request_stats['failed_requests'] += 1

            return APIResponse(
                success=False,
                data=None,
                error_message=str(e)
            )

    def search_multiple_platforms(self, platforms: List[Platform], query: str,
                                limit_per_platform: int = 25, **kwargs) -> AggregatedResult:
        """Search multiple platforms and aggregate results"""

        results = {}
        methods_used = {}
        errors = {}
        compliance_notes = []
        total_items = 0

        # Check if we should proceed based on compliance mode
        if self.compliance_mode == ComplianceMode.API_ONLY:
            compliance_notes.append("Operating in API-only mode - fallback methods disabled")

        # Search each platform
        for platform in platforms:
            try:
                result = self.search_single_platform(
                    platform, query, limit_per_platform, **kwargs
                )

                if result.success:
                    results[platform.value] = result.data
                    methods_used[platform.value] = result.method_used

                    # Count items
                    if isinstance(result.data, dict):
                        if 'posts' in result.data:
                            total_items += len(result.data['posts'])
                        elif 'videos' in result.data:
                            total_items += len(result.data['videos'])
                        elif 'count' in result.data:
                            total_items += result.data['count']
                else:
                    errors[platform.value] = result.error_message

            except Exception as e:
                errors[platform.value] = str(e)
                self.logger.error(f"Failed to search {platform.value}: {e}")

        return AggregatedResult(
            success=len(results) > 0,
            results=results,
            total_items=total_items,
            platforms_used=list(results.keys()),
            methods_used=methods_used,
            errors=errors,
            compliance_notes=compliance_notes
        )

    def search_with_fallback_strategy(self, search_requests: List[SearchRequest]) -> AggregatedResult:
        """Execute search requests with intelligent fallback strategy"""

        # Sort by priority
        sorted_requests = sorted(search_requests, key=lambda x: x.priority)

        results = {}
        methods_used = {}
        errors = {}
        compliance_notes = []
        total_items = 0

        for request in sorted_requests:
            platform = request.platform

            # Check if platform is available
            if platform not in self.clients:
                errors[platform.value] = "Platform not supported"
                continue

            try:
                # First try: API approach
                result = self.search_single_platform(
                    platform, request.query, request.limit, **(request.filters or {})
                )

                if result.success:
                    results[platform.value] = result.data
                    methods_used[platform.value] = result.method_used

                    # Count items
                    if isinstance(result.data, dict):
                        if 'posts' in result.data:
                            total_items += len(result.data['posts'])
                        elif 'videos' in result.data:
                            total_items += len(result.data['videos'])

                elif self._should_use_stealth_fallback(platform, result.error_message):
                    # Stealth fallback (only if compliance allows)
                    stealth_result = self._stealth_search(platform, request)
                    if stealth_result.success:
                        results[platform.value] = stealth_result.data
                        methods_used[platform.value] = "stealth_scraper"
                        compliance_notes.append(f"Used stealth scraper for {platform.value}")
                    else:
                        errors[platform.value] = stealth_result.error_message
                else:
                    errors[platform.value] = result.error_message

            except Exception as e:
                errors[platform.value] = str(e)
                self.logger.error(f"Failed to process {platform.value}: {e}")

        return AggregatedResult(
            success=len(results) > 0,
            results=results,
            total_items=total_items,
            platforms_used=list(results.keys()),
            methods_used=methods_used,
            errors=errors,
            compliance_notes=compliance_notes
        )

    def _should_use_stealth_fallback(self, platform: Platform, error_message: str) -> bool:
        """Determine if stealth fallback should be used"""

        # Never use stealth in strict compliance modes
        if self.compliance_mode in [ComplianceMode.API_ONLY]:
            return False

        # Only use for specific error types
        stealth_triggers = [
            "rate limit", "429", "blocked", "api limit exceeded",
            "quota exceeded", "no available methods"
        ]

        if error_message:
            return any(trigger in error_message.lower() for trigger in stealth_triggers)

        return False

    def _stealth_search(self, platform: Platform, request: SearchRequest) -> APIResponse:
        """Perform stealth search as last resort"""

        if self.compliance_mode == ComplianceMode.API_ONLY:
            return APIResponse(
                success=False,
                data=None,
                error_message="Stealth scraping disabled in API-only mode"
            )

        try:
            # Initialize stealth scraper for platform
            if platform.value not in self.stealth_scrapers:
                self.stealth_scrapers[platform.value] = AdvancedStealthScraper(platform.value)

            scraper = self.stealth_scrapers[platform.value]

            # Implement platform-specific stealth search
            if platform == Platform.REDDIT:
                return self._stealth_reddit_search(scraper, request)
            elif platform == Platform.YOUTUBE:
                return self._stealth_youtube_search(scraper, request)
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"Stealth search not implemented for {platform.value}"
                )

        except Exception as e:
            self.logger.error(f"Stealth search failed for {platform.value}: {e}")
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Stealth search error: {e}"
            )

    def _stealth_reddit_search(self, scraper: AdvancedStealthScraper,
                              request: SearchRequest) -> APIResponse:
        """Perform stealth Reddit search"""

        try:
            # Navigate to Reddit search
            search_url = f"https://old.reddit.com/search?q={request.query}&limit={request.limit}"
            scraper.human_like_navigation(search_url)

            # Extract search results
            selectors = {
                'titles': '.search-result .search-result-header a',
                'authors': '.search-result .author',
                'scores': '.search-result .score',
                'comments': '.search-result .search-comments',
                'subreddits': '.search-result .subreddit'
            }

            extracted_data = scraper.extract_page_data(selectors)

            # Format results
            posts = []
            if extracted_data.get('titles'):
                titles = extracted_data['titles'] if isinstance(extracted_data['titles'], list) else [extracted_data['titles']]

                for i, title in enumerate(titles):
                    post = {
                        'title': title,
                        'author': extracted_data.get('authors', [])[i] if isinstance(extracted_data.get('authors'), list) and i < len(extracted_data['authors']) else 'unknown',
                        'score': extracted_data.get('scores', [])[i] if isinstance(extracted_data.get('scores'), list) and i < len(extracted_data['scores']) else 0,
                        'subreddit': extracted_data.get('subreddits', [])[i] if isinstance(extracted_data.get('subreddits'), list) and i < len(extracted_data['subreddits']) else 'unknown',
                        'scraped_with': 'stealth_browser'
                    }
                    posts.append(post)

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'query': request.query,
                    'posts': posts,
                    'count': len(posts),
                    'detection_stats': scraper.get_detection_stats()
                },
                method_used="stealth_scraper"
            )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Stealth Reddit search failed: {e}"
            )

    def _stealth_youtube_search(self, scraper: AdvancedStealthScraper,
                               request: SearchRequest) -> APIResponse:
        """Perform stealth YouTube search"""

        try:
            # Navigate to YouTube search
            search_url = f"https://www.youtube.com/results?search_query={request.query}"
            scraper.human_like_navigation(search_url)

            # Wait for results to load
            time.sleep(3)

            # Extract video results
            selectors = {
                'titles': 'a#video-title',
                'channels': 'a.yt-simple-endpoint.style-scope.yt-formatted-string',
                'views': 'span.style-scope.ytd-video-meta-block:nth-child(1)',
                'durations': 'span.style-scope.ytd-thumbnail-overlay-time-status-renderer',
                'thumbnails': 'img.style-scope.yt-img-shadow'
            }

            extracted_data = scraper.extract_page_data(selectors)

            # Format results
            videos = []
            if extracted_data.get('titles'):
                titles = extracted_data['titles'] if isinstance(extracted_data['titles'], list) else [extracted_data['titles']]

                for i, title in enumerate(titles):
                    if i >= request.limit:
                        break

                    video = {
                        'title': title,
                        'channel': extracted_data.get('channels', [])[i] if isinstance(extracted_data.get('channels'), list) and i < len(extracted_data['channels']) else 'unknown',
                        'views': extracted_data.get('views', [])[i] if isinstance(extracted_data.get('views'), list) and i < len(extracted_data['views']) else 'unknown',
                        'duration': extracted_data.get('durations', [])[i] if isinstance(extracted_data.get('durations'), list) and i < len(extracted_data['durations']) else 'unknown',
                        'scraped_with': 'stealth_browser'
                    }
                    videos.append(video)

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'query': request.query,
                    'videos': videos,
                    'count': len(videos),
                    'detection_stats': scraper.get_detection_stats()
                },
                method_used="stealth_scraper"
            )

        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                error_message=f"Stealth YouTube search failed: {e}"
            )

    def get_platform_stats(self, platform: Platform) -> Dict:
        """Get statistics for a specific platform"""
        if platform not in self.clients:
            return {"error": "Platform not supported"}

        client = self.clients[platform]
        return client.get_stats()

    def get_global_stats(self) -> Dict:
        """Get global statistics across all platforms"""
        stats = dict(self.request_stats)

        # Add per-platform stats
        platform_stats = {}
        for platform, client in self.clients.items():
            platform_stats[platform.value] = client.get_stats()

        stats['platform_stats'] = platform_stats
        stats['compliance_mode'] = self.compliance_mode.value
        stats['global_rate_limiter'] = self.global_rate_limiter.get_status()

        return stats

    def cleanup(self):
        """Cleanup resources"""
        # Cleanup stealth scrapers
        for scraper in self.stealth_scrapers.values():
            try:
                scraper.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up stealth scraper: {e}")

        self.stealth_scrapers.clear()


class GlobalRateLimiter:
    """Global rate limiter across all platforms"""

    def __init__(self):
        self.requests_per_hour = 1000  # Global limit
        self.request_timestamps = {}
        self.platform_limits = {
            'reddit': 100,
            'youtube': 200,
            'twitter': 150,
            'instagram': 50
        }

    def can_make_request(self, platform: str) -> bool:
        """Check if request can be made"""
        now = time.time()
        hour_ago = now - 3600

        # Clean old timestamps
        if platform not in self.request_timestamps:
            self.request_timestamps[platform] = []

        self.request_timestamps[platform] = [
            ts for ts in self.request_timestamps[platform] if ts > hour_ago
        ]

        # Check platform-specific limit
        platform_limit = self.platform_limits.get(platform, 100)
        if len(self.request_timestamps[platform]) >= platform_limit:
            return False

        # Check global limit
        total_recent_requests = sum(
            len(timestamps) for timestamps in self.request_timestamps.values()
        )

        return total_recent_requests < self.requests_per_hour

    def record_request(self, platform: str):
        """Record a request"""
        if platform not in self.request_timestamps:
            self.request_timestamps[platform] = []

        self.request_timestamps[platform].append(time.time())

    def get_wait_time(self, platform: str) -> float:
        """Get recommended wait time"""
        if not self.request_timestamps.get(platform):
            return 0

        # Find oldest request in current hour
        hour_ago = time.time() - 3600
        recent_requests = [
            ts for ts in self.request_timestamps[platform] if ts > hour_ago
        ]

        if not recent_requests:
            return 0

        platform_limit = self.platform_limits.get(platform, 100)
        if len(recent_requests) >= platform_limit:
            # Wait until oldest request is over an hour old
            oldest_request = min(recent_requests)
            return max(0, 3600 - (time.time() - oldest_request))

        return 60  # Default 1 minute wait

    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        now = time.time()
        hour_ago = now - 3600

        status = {}
        for platform, timestamps in self.request_timestamps.items():
            recent_count = len([ts for ts in timestamps if ts > hour_ago])
            limit = self.platform_limits.get(platform, 100)

            status[platform] = {
                'requests_last_hour': recent_count,
                'limit': limit,
                'remaining': max(0, limit - recent_count)
            }

        return status