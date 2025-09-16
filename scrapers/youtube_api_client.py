# scrapers/youtube_api_client.py
"""
YouTube API client with compliant fallback methods
Implements YouTube Data API v3 with RSS and public page fallbacks
"""

import time
import json
import requests
import feedparser
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote, urlparse, parse_qs
import os
import logging
import re
from bs4 import BeautifulSoup

from .base_api_client import BaseAPIClient, APIResponse, APIException, load_platform_config


class YouTubeAPIClient(BaseAPIClient):
    """YouTube API client with compliant fallback methods"""

    def __init__(self):
        config = load_platform_config('youtube')
        super().__init__('youtube', config)

        # YouTube API base URL
        self.api_base = "https://www.googleapis.com/youtube/v3"

        # Track quota usage
        self.quota_used = 0
        self.daily_quota_limit = self.config.get('api', {}).get('rate_limits', {}).get('daily_quota', 10000)

    def _has_api_credentials(self) -> bool:
        """Check if YouTube API key is available"""
        return bool(os.getenv('YOUTUBE_API_KEY'))

    def _get_auth_headers(self) -> Dict[str, str]:
        """YouTube API uses key parameter, not headers"""
        return {}

    def search_videos(self, query: str, limit: int = 25, **kwargs) -> APIResponse:
        """Search for videos on YouTube"""
        return self.search(query, limit, **kwargs)

    def get_video_details(self, video_id: str) -> APIResponse:
        """Get detailed information about a video"""

        if self._should_use_api():
            try:
                return self._api_get_video_details(video_id)
            except Exception as e:
                self.logger.warning(f"API method failed: {e}")

        if self._should_use_fallback():
            return self._fallback_get_video_details(video_id)

        return APIResponse(
            success=False,
            data=None,
            error_message="No available methods for video details"
        )

    def get_channel_videos(self, channel_id: str, limit: int = 25) -> APIResponse:
        """Get videos from a specific channel"""

        if self._should_use_api():
            try:
                return self._api_get_channel_videos(channel_id, limit)
            except Exception as e:
                self.logger.warning(f"API method failed: {e}")

        if self._should_use_fallback():
            return self._fallback_get_channel_videos(channel_id, limit)

        return APIResponse(
            success=False,
            data=None,
            error_message="No available methods for channel videos"
        )

    def _api_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search YouTube using official API"""

        if not self._check_quota_available(100):  # Search costs 100 units
            raise APIException("Daily quota exceeded")

        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': min(limit, 50),  # YouTube API limit
            'order': kwargs.get('order', 'relevance'),
            'publishedAfter': kwargs.get('published_after'),
            'publishedBefore': kwargs.get('published_before'),
            'videoDuration': kwargs.get('duration'),
            'key': os.getenv('YOUTUBE_API_KEY')
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = self._make_api_request(f"{self.api_base}/search", params)
            data = response.json()

            self.quota_used += 100  # Record quota usage

            videos = []
            for item in data.get('items', []):
                videos.append(self._format_api_video(item))

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'query': query,
                    'videos': videos,
                    'count': len(videos),
                    'next_page_token': data.get('nextPageToken'),
                    'total_results': data.get('pageInfo', {}).get('totalResults', 0)
                },
                method_used="youtube_api"
            )

        except Exception as e:
            raise APIException(f"YouTube API search failed: {e}")

    def _api_get_video_details(self, video_id: str) -> APIResponse:
        """Get video details using YouTube API"""

        if not self._check_quota_available(1):
            raise APIException("Daily quota exceeded")

        params = {
            'part': 'snippet,statistics,contentDetails',
            'id': video_id,
            'key': os.getenv('YOUTUBE_API_KEY')
        }

        try:
            response = self._make_api_request(f"{self.api_base}/videos", params)
            data = response.json()

            self.quota_used += 1

            if not data.get('items'):
                raise APIException(f"Video {video_id} not found")

            video_data = self._format_detailed_video(data['items'][0])

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'video': video_data
                },
                method_used="youtube_api"
            )

        except Exception as e:
            raise APIException(f"YouTube API video details failed: {e}")

    def _api_get_channel_videos(self, channel_id: str, limit: int) -> APIResponse:
        """Get channel videos using YouTube API"""

        if not self._check_quota_available(1):  # Channel info costs 1 unit
            raise APIException("Daily quota exceeded")

        # First get channel uploads playlist ID
        params = {
            'part': 'contentDetails',
            'id': channel_id,
            'key': os.getenv('YOUTUBE_API_KEY')
        }

        try:
            response = self._make_api_request(f"{self.api_base}/channels", params)
            channel_data = response.json()

            if not channel_data.get('items'):
                raise APIException(f"Channel {channel_id} not found")

            uploads_playlist_id = channel_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Get videos from uploads playlist
            if not self._check_quota_available(1):  # Playlist items costs 1 unit
                raise APIException("Daily quota exceeded")

            params = {
                'part': 'snippet',
                'playlistId': uploads_playlist_id,
                'maxResults': min(limit, 50),
                'key': os.getenv('YOUTUBE_API_KEY')
            }

            response = self._make_api_request(f"{self.api_base}/playlistItems", params)
            data = response.json()

            self.quota_used += 2  # 1 for channel, 1 for playlist

            videos = []
            for item in data.get('items', []):
                videos.append(self._format_playlist_video(item))

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'channel_id': channel_id,
                    'videos': videos,
                    'count': len(videos)
                },
                method_used="youtube_api"
            )

        except Exception as e:
            raise APIException(f"YouTube API channel videos failed: {e}")

    def _rss_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search using YouTube RSS feeds (limited functionality)"""

        # RSS feeds are primarily for channel videos, not general search
        # This is a fallback for getting recent videos from popular channels

        try:
            # Try to find channels related to the query using web search approach
            # This is a simplified implementation
            videos = []

            # For demonstration, we'll check a few popular channels
            # In practice, you'd have a more sophisticated channel discovery method
            popular_channels = [
                'UCBJycsmduvYEL83R_U4JriQ',  # Marques Brownlee
                'UC2eYFnH61tmytImy1mTYvhA',  # Luke Smith
                # Add more relevant channels based on query
            ]

            for channel_id in popular_channels[:3]:  # Limit to avoid too many requests
                try:
                    channel_videos = self._rss_get_channel_videos(channel_id, limit=10)
                    if channel_videos.success:
                        # Filter videos by query relevance
                        for video in channel_videos.data['videos']:
                            if query.lower() in video['title'].lower() or query.lower() in video.get('description', '').lower():
                                videos.append(video)
                                if len(videos) >= limit:
                                    break
                        if len(videos) >= limit:
                            break
                except:
                    continue

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'query': query,
                    'videos': videos[:limit],
                    'count': len(videos[:limit]),
                    'note': 'RSS fallback provides limited search functionality'
                },
                method_used="rss_fallback"
            )

        except Exception as e:
            self.logger.error(f"RSS search failed: {e}")
            raise

    def _rss_get_channel_videos(self, channel_id: str, limit: int = 25) -> APIResponse:
        """Get channel videos using RSS feed"""

        try:
            # YouTube channel RSS feed URL
            rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

            # Add delay for respectful scraping
            time.sleep(1)

            response = requests.get(rss_url, timeout=30)
            response.raise_for_status()

            # Parse RSS feed
            feed = feedparser.parse(response.content)

            videos = []
            for entry in feed.entries[:limit]:
                video_id = entry.yt_videoid if hasattr(entry, 'yt_videoid') else self._extract_video_id_from_url(entry.link)

                video_data = {
                    'id': video_id,
                    'title': entry.title,
                    'description': entry.summary if hasattr(entry, 'summary') else '',
                    'published_at': entry.published if hasattr(entry, 'published') else '',
                    'channel_id': channel_id,
                    'channel_title': entry.author if hasattr(entry, 'author') else '',
                    'url': entry.link,
                    'thumbnail': entry.media_thumbnail[0]['url'] if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail else None
                }

                videos.append(video_data)

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'channel_id': channel_id,
                    'videos': videos,
                    'count': len(videos)
                },
                method_used="rss_feed"
            )

        except Exception as e:
            self.logger.error(f"RSS channel fetch failed: {e}")
            raise

    def _fallback_get_video_details(self, video_id: str) -> APIResponse:
        """Get video details by parsing video page"""

        try:
            url = f"https://www.youtube.com/watch?v={video_id}"

            # Add delay for respectful scraping
            time.sleep(2)

            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract basic video info from page title and meta tags
            title = soup.find('title')
            title_text = title.text.replace(' - YouTube', '') if title else 'Unknown'

            # Try to find JSON data in script tags
            video_data = {
                'id': video_id,
                'title': title_text,
                'url': url,
                'description': '',
                'view_count': None,
                'like_count': None,
                'channel_title': '',
                'published_at': ''
            }

            # Look for embedded JSON data
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'ytInitialData' in script.string:
                    # Try to extract structured data
                    # This is simplified - YouTube's JS structure is complex
                    pass

            # Extract meta tags
            description_meta = soup.find('meta', {'name': 'description'})
            if description_meta:
                video_data['description'] = description_meta.get('content', '')

            return APIResponse(
                success=True,
                data={
                    'platform': 'youtube',
                    'video': video_data,
                    'note': 'Fallback method provides limited data'
                },
                method_used="web_scraping"
            )

        except Exception as e:
            self.logger.error(f"Fallback video details failed: {e}")
            raise

    def _check_quota_available(self, cost: int) -> bool:
        """Check if enough quota is available for the request"""
        return (self.quota_used + cost) <= self.daily_quota_limit

    def _extract_video_id_from_url(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        parsed = urlparse(url)
        if parsed.hostname in ['youtube.com', 'www.youtube.com']:
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [''])[0]
        elif parsed.hostname in ['youtu.be']:
            return parsed.path[1:]
        return ''

    def _format_api_video(self, item: Dict) -> Dict:
        """Format API search result to standard format"""
        snippet = item['snippet']
        return {
            'id': item['id']['videoId'],
            'title': snippet['title'],
            'description': snippet['description'],
            'published_at': snippet['publishedAt'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'thumbnail': snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }

    def _format_detailed_video(self, item: Dict) -> Dict:
        """Format detailed video data from API"""
        snippet = item['snippet']
        statistics = item.get('statistics', {})
        content_details = item.get('contentDetails', {})

        return {
            'id': item['id'],
            'title': snippet['title'],
            'description': snippet['description'],
            'published_at': snippet['publishedAt'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'thumbnail': snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
            'url': f"https://www.youtube.com/watch?v={item['id']}",
            'duration': content_details.get('duration'),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'tags': snippet.get('tags', [])
        }

    def _format_playlist_video(self, item: Dict) -> Dict:
        """Format playlist video data from API"""
        snippet = item['snippet']
        return {
            'id': snippet['resourceId']['videoId'],
            'title': snippet['title'],
            'description': snippet['description'],
            'published_at': snippet['publishedAt'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'thumbnail': snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
            'url': f"https://www.youtube.com/watch?v={snippet['resourceId']['videoId']}"
        }

    def get_quota_usage(self) -> Dict:
        """Get current quota usage"""
        return {
            'quota_used': self.quota_used,
            'daily_limit': self.daily_quota_limit,
            'remaining': self.daily_quota_limit - self.quota_used,
            'percentage_used': (self.quota_used / self.daily_quota_limit) * 100
        }