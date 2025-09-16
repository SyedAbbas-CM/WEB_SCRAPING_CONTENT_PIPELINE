# scrapers/reddit_api_client.py
"""
Reddit API client with compliant fallback methods
Implements Reddit API access with ethical fallback scraping
"""

import time
import json
import requests
import praw
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote
import os
import logging

from .base_api_client import BaseAPIClient, APIResponse, APIException, load_platform_config


class RedditAPIClient(BaseAPIClient):
    """Reddit API client with compliant fallback methods"""

    def __init__(self):
        config = load_platform_config('reddit')
        super().__init__('reddit', config)

        # Initialize PRAW client if credentials available
        self.reddit_client = None
        if self._has_api_credentials():
            try:
                self.reddit_client = praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                    user_agent=self.config.get('api', {}).get('user_agent', 'ScrapeHive/1.0')
                )
                self.logger.info("Reddit API client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Reddit API client: {e}")

    def _has_api_credentials(self) -> bool:
        """Check if Reddit API credentials are available"""
        return bool(
            os.getenv('REDDIT_CLIENT_ID') and
            os.getenv('REDDIT_CLIENT_SECRET')
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Reddit API"""
        # PRAW handles authentication internally
        return {}

    def search_subreddit(self, subreddit: str, query: str = None, sort: str = 'hot',
                        limit: int = 25, time_filter: str = 'all') -> APIResponse:
        """Search or list posts in a subreddit"""

        if query:
            return self.search(f"subreddit:{subreddit} {query}", limit=limit)
        else:
            return self.get_subreddit_posts(subreddit, sort, limit, time_filter)

    def get_subreddit_posts(self, subreddit: str, sort: str = 'hot',
                           limit: int = 25, time_filter: str = 'all') -> APIResponse:
        """Get posts from a subreddit"""

        if self._should_use_api() and self.reddit_client:
            try:
                return self._api_get_subreddit_posts(subreddit, sort, limit, time_filter)
            except Exception as e:
                self.logger.warning(f"API method failed: {e}")

        if self._should_use_fallback():
            return self._fallback_get_subreddit_posts(subreddit, sort, limit, time_filter)

        return APIResponse(
            success=False,
            data=None,
            error_message="No available methods for subreddit posts"
        )

    def get_post_details(self, post_id: str, subreddit: str = None) -> APIResponse:
        """Get post details with comments"""

        if self._should_use_api() and self.reddit_client:
            try:
                return self._api_get_post_details(post_id)
            except Exception as e:
                self.logger.warning(f"API method failed: {e}")

        if self._should_use_fallback():
            return self._fallback_get_post_details(post_id, subreddit)

        return APIResponse(
            success=False,
            data=None,
            error_message="No available methods for post details"
        )

    def _api_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search Reddit using official API"""

        if not self.reddit_client:
            raise APIException("Reddit API client not initialized")

        try:
            # Use PRAW to search
            submissions = list(self.reddit_client.subreddit("all").search(
                query,
                limit=limit,
                sort=kwargs.get('sort', 'relevance'),
                time_filter=kwargs.get('time_filter', 'all')
            ))

            posts = []
            for submission in submissions:
                posts.append(self._format_submission(submission))

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'query': query,
                    'posts': posts,
                    'count': len(posts)
                },
                method_used="reddit_api"
            )

        except Exception as e:
            raise APIException(f"Reddit API search failed: {e}")

    def _api_get_subreddit_posts(self, subreddit: str, sort: str, limit: int, time_filter: str) -> APIResponse:
        """Get subreddit posts using Reddit API"""

        try:
            sub = self.reddit_client.subreddit(subreddit)

            if sort == 'hot':
                submissions = sub.hot(limit=limit)
            elif sort == 'new':
                submissions = sub.new(limit=limit)
            elif sort == 'top':
                submissions = sub.top(time_filter=time_filter, limit=limit)
            elif sort == 'rising':
                submissions = sub.rising(limit=limit)
            else:
                submissions = sub.hot(limit=limit)

            posts = []
            for submission in submissions:
                posts.append(self._format_submission(submission))

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'subreddit': subreddit,
                    'sort': sort,
                    'posts': posts,
                    'count': len(posts)
                },
                method_used="reddit_api"
            )

        except Exception as e:
            raise APIException(f"Reddit API subreddit fetch failed: {e}")

    def _api_get_post_details(self, post_id: str) -> APIResponse:
        """Get post details using Reddit API"""

        try:
            submission = self.reddit_client.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Don't expand all comments for rate limiting

            post_data = self._format_submission(submission)

            comments = []
            for comment in submission.comments.list()[:50]:  # Limit comments
                if hasattr(comment, 'body'):
                    comments.append(self._format_comment(comment))

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'post': post_data,
                    'comments': comments,
                    'comment_count': len(comments)
                },
                method_used="reddit_api"
            )

        except Exception as e:
            raise APIException(f"Reddit API post fetch failed: {e}")

    def _json_endpoint_search(self, query: str, limit: int, **kwargs) -> APIResponse:
        """Search using Reddit's JSON endpoints"""

        try:
            # Use Reddit's search JSON endpoint
            params = {
                'q': query,
                'limit': min(limit, 100),  # Reddit's limit
                'sort': kwargs.get('sort', 'relevance'),
                't': kwargs.get('time_filter', 'all'),
                'raw_json': 1
            }

            url = "https://www.reddit.com/search.json"

            response = self._make_json_request(url, params)
            data = response.json()

            posts = []
            if 'data' in data and 'children' in data['data']:
                for child in data['data']['children']:
                    if child['kind'] == 't3':  # Link/post
                        posts.append(self._format_json_post(child['data']))

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'query': query,
                    'posts': posts,
                    'count': len(posts)
                },
                method_used="json_endpoint"
            )

        except Exception as e:
            self.logger.error(f"JSON endpoint search failed: {e}")
            raise

    def _fallback_get_subreddit_posts(self, subreddit: str, sort: str, limit: int, time_filter: str) -> APIResponse:
        """Get subreddit posts using JSON fallback"""

        try:
            # Use Reddit's JSON endpoint for subreddit
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {
                'limit': min(limit, 100),
                'raw_json': 1
            }

            if sort == 'top':
                params['t'] = time_filter

            response = self._make_json_request(url, params)
            data = response.json()

            posts = []
            if 'data' in data and 'children' in data['data']:
                for child in data['data']['children']:
                    if child['kind'] == 't3':
                        posts.append(self._format_json_post(child['data']))

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'subreddit': subreddit,
                    'sort': sort,
                    'posts': posts,
                    'count': len(posts)
                },
                method_used="json_fallback"
            )

        except Exception as e:
            self.logger.error(f"Fallback subreddit fetch failed: {e}")
            raise

    def _fallback_get_post_details(self, post_id: str, subreddit: str = None) -> APIResponse:
        """Get post details using JSON fallback"""

        if not subreddit:
            # Try to get post info first to find subreddit
            try:
                url = f"https://www.reddit.com/api/info.json"
                params = {'id': f't3_{post_id}', 'raw_json': 1}
                response = self._make_json_request(url, params)
                data = response.json()

                if data['data']['children']:
                    subreddit = data['data']['children'][0]['data']['subreddit']
                else:
                    raise Exception("Post not found")

            except Exception as e:
                raise Exception(f"Could not determine subreddit for post {post_id}: {e}")

        try:
            # Get post with comments
            url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            params = {'raw_json': 1, 'limit': 50}  # Limit comments

            response = self._make_json_request(url, params)
            data = response.json()

            if len(data) < 2:
                raise Exception("Invalid post data structure")

            # First element is post, second is comments
            post_data = data[0]['data']['children'][0]['data']
            post = self._format_json_post(post_data)

            comments = []
            if len(data) > 1 and 'children' in data[1]['data']:
                self._extract_comments(data[1]['data']['children'], comments)

            return APIResponse(
                success=True,
                data={
                    'platform': 'reddit',
                    'post': post,
                    'comments': comments,
                    'comment_count': len(comments)
                },
                method_used="json_fallback"
            )

        except Exception as e:
            self.logger.error(f"Fallback post fetch failed: {e}")
            raise

    def _make_json_request(self, url: str, params: Dict) -> requests.Response:
        """Make request to Reddit JSON endpoints with proper headers and delays"""

        # Apply delay for respectful scraping
        delay_config = self.config.get('fallback', {}).get('delays', {})
        delay = delay_config.get('min', 2)
        time.sleep(delay)

        headers = {
            'User-Agent': self._get_user_agent(),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        response = self.session.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code == 429:
            self.logger.warning("Rate limited by Reddit")
            raise APIException("Rate limited")
        elif response.status_code != 200:
            raise APIException(f"HTTP {response.status_code}")

        return response

    def _format_submission(self, submission) -> Dict:
        """Format PRAW submission object to standard format"""
        return {
            'id': submission.id,
            'title': submission.title,
            'author': str(submission.author) if submission.author else '[deleted]',
            'created_utc': submission.created_utc,
            'score': submission.score,
            'upvote_ratio': getattr(submission, 'upvote_ratio', None),
            'num_comments': submission.num_comments,
            'permalink': f"https://reddit.com{submission.permalink}",
            'url': submission.url,
            'selftext': getattr(submission, 'selftext', ''),
            'subreddit': submission.subreddit.display_name,
            'is_video': getattr(submission, 'is_video', False),
            'nsfw': submission.over_18,
            'stickied': submission.stickied,
            'locked': submission.locked,
            'flair': getattr(submission, 'link_flair_text', None),
            'domain': getattr(submission, 'domain', None)
        }

    def _format_comment(self, comment) -> Dict:
        """Format PRAW comment object to standard format"""
        return {
            'id': comment.id,
            'author': str(comment.author) if comment.author else '[deleted]',
            'body': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
            'parent_id': comment.parent_id,
            'depth': getattr(comment, 'depth', 0),
            'is_submitter': comment.is_submitter,
            'stickied': comment.stickied,
            'distinguished': comment.distinguished
        }

    def _format_json_post(self, post_data: Dict) -> Dict:
        """Format JSON post data to standard format"""
        return {
            'id': post_data['id'],
            'title': post_data['title'],
            'author': post_data.get('author', '[deleted]'),
            'created_utc': post_data['created_utc'],
            'score': post_data['score'],
            'upvote_ratio': post_data.get('upvote_ratio'),
            'num_comments': post_data['num_comments'],
            'permalink': f"https://reddit.com{post_data['permalink']}",
            'url': post_data['url'],
            'selftext': post_data.get('selftext', ''),
            'subreddit': post_data['subreddit'],
            'is_video': post_data.get('is_video', False),
            'nsfw': post_data['over_18'],
            'stickied': post_data['stickied'],
            'locked': post_data.get('locked', False),
            'flair': post_data.get('link_flair_text'),
            'domain': post_data.get('domain')
        }

    def _extract_comments(self, comments_data: List, comments_list: List, depth: int = 0):
        """Recursively extract comments from JSON data"""
        for comment_item in comments_data:
            if comment_item['kind'] == 't1':  # Comment
                comment_data = comment_item['data']

                formatted_comment = {
                    'id': comment_data['id'],
                    'author': comment_data.get('author', '[deleted]'),
                    'body': comment_data.get('body', '[removed]'),
                    'score': comment_data['score'],
                    'created_utc': comment_data['created_utc'],
                    'parent_id': comment_data['parent_id'],
                    'depth': depth,
                    'stickied': comment_data.get('stickied', False),
                    'distinguished': comment_data.get('distinguished')
                }

                comments_list.append(formatted_comment)

                # Process replies
                if comment_data.get('replies') and isinstance(comment_data['replies'], dict):
                    self._extract_comments(
                        comment_data['replies']['data']['children'],
                        comments_list,
                        depth + 1
                    )
            elif comment_item['kind'] == 'more':
                # Skip "load more comments" items for simplicity
                continue