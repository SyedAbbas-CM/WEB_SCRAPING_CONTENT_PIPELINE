# scrapers/reddit_scraper.py
"""
Enhanced Reddit Scraper with API and anti-ban measures
Uses both official API and web scraping with proper rate limiting
"""

import time
import json
import random
import requests
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse, quote
import praw
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from utils.reddit_auth_manager import RedditAuthManager
from utils.ip_manager import IPManager, IPType
from utils.rate_limiter import EnhancedRateLimiter

class RedditScraper:
    """
    Advanced Reddit scraper with multiple methods:
    1. Official Reddit API (authenticated)
    2. JSON endpoints (no auth)
    3. Old Reddit (web scraping)
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'reddit_scraper.{node_id}')
        
        # Initialize managers
        self.auth_manager = RedditAuthManager()
        self.ip_manager = IPManager()
        self.rate_limiter = EnhancedRateLimiter('reddit')
        
        # User agent generator
        self.ua = UserAgent()
        
        # Scraping methods in order of preference
        self.methods = ['api', 'json', 'old_reddit']
        
        # Session management
        self.sessions = {}
        
        # Stats
        self.stats = {
            'api_calls': 0,
            'json_calls': 0,
            'web_scrapes': 0,
            'blocks_encountered': 0,
            'ip_rotations': 0
        }
        
    def scrape(self, url: str, params: Dict = None) -> Dict:
        """
        Main scraping method - tries multiple approaches
        """
        params = params or {}
        
        # Parse URL
        parsed = self._parse_reddit_url(url)
        
        # Try each method in order
        for method in self.methods:
            try:
                if method == 'api' and self.auth_manager.accounts:
                    return self._scrape_via_api(parsed, params)
                elif method == 'json':
                    return self._scrape_via_json(parsed, params)
                elif method == 'old_reddit':
                    return self._scrape_via_web(parsed, params)
                    
            except RateLimitException:
                self.logger.warning(f"Rate limit hit for method: {method}")
                continue
            except BlockedException:
                self.logger.warning(f"Blocked on method: {method}")
                self.stats['blocks_encountered'] += 1
                # Rotate IP and try next method
                self._handle_block()
                continue
            except Exception as e:
                self.logger.error(f"Error with method {method}: {e}")
                continue
        
        raise Exception("All scraping methods failed")
    
    def _parse_reddit_url(self, url: str) -> Dict:
        """Parse Reddit URL to extract components"""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        result = {
            'url': url,
            'type': 'unknown',
            'subreddit': None,
            'post_id': None,
            'username': None,
            'params': {}
        }
        
        # Detect URL type
        if len(path_parts) >= 2:
            if path_parts[0] == 'r' and len(path_parts) == 2:
                result['type'] = 'subreddit'
                result['subreddit'] = path_parts[1]
            elif path_parts[0] == 'r' and path_parts[2] == 'comments':
                result['type'] = 'post'
                result['subreddit'] = path_parts[1]
                result['post_id'] = path_parts[3] if len(path_parts) > 3 else None
            elif path_parts[0] in ['u', 'user']:
                result['type'] = 'user'
                result['username'] = path_parts[1]
        
        return result
    
    def _scrape_via_api(self, parsed: Dict, params: Dict) -> Dict:
        """Scrape using official Reddit API"""
        reddit = self.auth_manager.get_reddit_instance()
        if not reddit:
            raise Exception("No Reddit API instance available")
        
        # Check rate limit
        if not self.rate_limiter.check_limit('api'):
            raise RateLimitException("API rate limit exceeded")
        
        self.stats['api_calls'] += 1
        
        try:
            if parsed['type'] == 'subreddit':
                return self._api_scrape_subreddit(reddit, parsed['subreddit'], params)
            elif parsed['type'] == 'post':
                return self._api_scrape_post(reddit, parsed['post_id'])
            elif parsed['type'] == 'user':
                return self._api_scrape_user(reddit, parsed['username'], params)
            else:
                raise ValueError(f"Unknown URL type: {parsed['type']}")
                
        except praw.exceptions.APIException as e:
            if e.error_type == 'THREAD_LOCKED':
                self.logger.warning("Thread is locked")
            elif e.error_type == 'SUBREDDIT_NOEXIST':
                self.logger.warning("Subreddit does not exist")
            else:
                raise
                
    def _api_scrape_subreddit(self, reddit: praw.Reddit, subreddit_name: str, params: Dict) -> Dict:
        """Scrape subreddit using API"""
        subreddit = reddit.subreddit(subreddit_name)
        
        sort = params.get('sort', 'hot')
        limit = params.get('limit', 25)
        time_filter = params.get('time_filter', 'all')
        
        posts = []
        
        # Get posts based on sort method
        if sort == 'hot':
            submissions = subreddit.hot(limit=limit)
        elif sort == 'new':
            submissions = subreddit.new(limit=limit)
        elif sort == 'top':
            submissions = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort == 'rising':
            submissions = subreddit.rising(limit=limit)
        else:
            submissions = subreddit.hot(limit=limit)
        
        for submission in submissions:
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author) if submission.author else '[deleted]',
                'created_utc': submission.created_utc,
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'permalink': f"https://reddit.com{submission.permalink}",
                'url': submission.url,
                'selftext': submission.selftext,
                'is_video': submission.is_video,
                'subreddit': submission.subreddit.display_name,
                'awards': len(submission.all_awardings),
                'spoiler': submission.spoiler,
                'nsfw': submission.over_18,
                'stickied': submission.stickied,
                'locked': submission.locked,
                'flair': submission.link_flair_text
            }
            
            posts.append(post_data)
            
            # Rate limiting
            time.sleep(0.1)
        
        return {
            'platform': 'reddit',
            'type': 'subreddit',
            'subreddit': subreddit_name,
            'sort': sort,
            'posts': posts,
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'api'
        }
    
    def _api_scrape_post(self, reddit: praw.Reddit, post_id: str) -> Dict:
        """Scrape post and comments using API"""
        submission = reddit.submission(id=post_id)
        
        # Expand all comments
        submission.comments.replace_more(limit=None)
        
        # Extract post data
        post_data = {
            'id': submission.id,
            'title': submission.title,
            'author': str(submission.author) if submission.author else '[deleted]',
            'created_utc': submission.created_utc,
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'selftext': submission.selftext,
            'url': submission.url,
            'subreddit': submission.subreddit.display_name
        }
        
        # Extract comments
        comments = []
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.MoreComments):
                continue
                
            comment_data = {
                'id': comment.id,
                'author': str(comment.author) if comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'edited': comment.edited,
                'parent_id': comment.parent_id,
                'depth': comment.depth,
                'is_submitter': comment.is_submitter,
                'stickied': comment.stickied,
                'distinguished': comment.distinguished,
                'awards': len(comment.all_awardings)
            }
            
            comments.append(comment_data)
        
        return {
            'platform': 'reddit',
            'type': 'post',
            'post': post_data,
            'comments': comments,
            'comment_count': len(comments),
            'scraped_at': time.time(),
            'method': 'api'
        }
    
    def _scrape_via_json(self, parsed: Dict, params: Dict) -> Dict:
        """Scrape using Reddit's JSON endpoints (no auth required)"""
        
        # Check rate limit
        if not self.rate_limiter.check_limit('json'):
            raise RateLimitException("JSON endpoint rate limit exceeded")
        
        # Get IP for request
        ip_config = self.ip_manager.get_ip_for_platform('reddit')
        
        # Build URL
        if parsed['type'] == 'subreddit':
            sort = params.get('sort', 'hot')
            url = f"https://www.reddit.com/r/{parsed['subreddit']}/{sort}.json"
        elif parsed['type'] == 'post':
            url = f"https://www.reddit.com/r/{parsed['subreddit']}/comments/{parsed['post_id']}.json"
        elif parsed['type'] == 'user':
            url = f"https://www.reddit.com/user/{parsed['username']}/about.json"
        else:
            url = parsed['url'] + '.json'
        
        # Add parameters
        if params:
            url += '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
        
        # Make request
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        proxies = ip_config.to_proxy_dict() if ip_config and ip_config.port else None
        
        try:
            response = requests.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=30,
                allow_redirects=False
            )
            
            self.stats['json_calls'] += 1
            
            # Check for blocks
            if response.status_code == 429:
                self.logger.warning("Rate limited by Reddit")
                raise RateLimitException("429 from Reddit")
            elif response.status_code == 403:
                self.logger.warning("Forbidden - possible IP ban")
                if ip_config:
                    self.ip_manager.mark_ip_blocked(ip_config, 'reddit')
                raise BlockedException("403 from Reddit")
            elif response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            data = response.json()
            
            # Parse based on type
            if parsed['type'] == 'subreddit':
                return self._parse_json_subreddit(data, parsed['subreddit'])
            elif parsed['type'] == 'post':
                return self._parse_json_post(data)
            else:
                return {'data': data, 'method': 'json'}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"JSON request failed: {e}")
            raise
    
    def _scrape_via_web(self, parsed: Dict, params: Dict) -> Dict:
        """Scrape using old.reddit.com web interface"""
        
        # Check rate limit
        if not self.rate_limiter.check_limit('web'):
            raise RateLimitException("Web scraping rate limit exceeded")
        
        # Get IP
        ip_config = self.ip_manager.get_ip_for_platform('reddit')
        
        # Build URL (use old.reddit.com for easier parsing)
        base_url = parsed['url'].replace('www.reddit.com', 'old.reddit.com')
        base_url = base_url.replace('reddit.com', 'old.reddit.com')
        
        # Create session
        session = self._get_session('web', ip_config)
        
        try:
            response = session.get(base_url, timeout=30)
            
            self.stats['web_scrapes'] += 1
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for Reddit's "you broke reddit" page
            if "you broke reddit" in response.text.lower():
                raise BlockedException("Reddit block page detected")
            
            # Parse based on type
            if parsed['type'] == 'subreddit':
                return self._parse_web_subreddit(soup, parsed['subreddit'])
            elif parsed['type'] == 'post':
                return self._parse_web_post(soup)
            else:
                return self._parse_web_generic(soup)
                
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            raise
    
    def _get_session(self, session_type: str, ip_config) -> requests.Session:
        """Get or create session with proper configuration"""
        session_key = f"{session_type}_{ip_config.address if ip_config else 'direct'}"
        
        if session_key not in self.sessions:
            session = requests.Session()
            
            # Set headers
            session.headers.update({
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache'
            })
            
            # Set proxy
            if ip_config and ip_config.port:
                session.proxies.update(ip_config.to_proxy_dict())
            
            self.sessions[session_key] = session
        
        return self.sessions[session_key]
    
    def _parse_json_subreddit(self, data: Dict, subreddit: str) -> Dict:
        """Parse JSON response for subreddit"""
        posts = []
        
        if 'data' in data and 'children' in data['data']:
            for child in data['data']['children']:
                post = child['data']
                posts.append({
                    'id': post['id'],
                    'title': post['title'],
                    'author': post.get('author', '[deleted]'),
                    'created_utc': post['created_utc'],
                    'score': post['score'],
                    'num_comments': post['num_comments'],
                    'permalink': f"https://reddit.com{post['permalink']}",
                    'url': post['url'],
                    'selftext': post.get('selftext', ''),
                    'subreddit': post['subreddit']
                })
        
        return {
            'platform': 'reddit',
            'type': 'subreddit',
            'subreddit': subreddit,
            'posts': posts,
            'count': len(posts),
            'after': data['data'].get('after'),
            'scraped_at': time.time(),
            'method': 'json'
        }
    
    def _parse_json_post(self, data: List) -> Dict:
        """Parse JSON response for post with comments"""
        if len(data) < 2:
            raise ValueError("Invalid post data")
        
        # First element is the post
        post_data = data[0]['data']['children'][0]['data']
        
        post = {
            'id': post_data['id'],
            'title': post_data['title'],
            'author': post_data.get('author', '[deleted]'),
            'created_utc': post_data['created_utc'],
            'score': post_data['score'],
            'num_comments': post_data['num_comments'],
            'selftext': post_data.get('selftext', ''),
            'url': post_data['url'],
            'subreddit': post_data['subreddit']
        }
        
        # Second element contains comments
        comments = []
        
        def parse_comments(comment_data, depth=0):
            for child in comment_data:
                if child['kind'] == 't1':  # Comment
                    comment = child['data']
                    comments.append({
                        'id': comment['id'],
                        'author': comment.get('author', '[deleted]'),
                        'body': comment.get('body', '[removed]'),
                        'score': comment['score'],
                        'created_utc': comment['created_utc'],
                        'parent_id': comment['parent_id'],
                        'depth': depth
                    })
                    
                    # Parse replies
                    if comment.get('replies') and isinstance(comment['replies'], dict):
                        parse_comments(comment['replies']['data']['children'], depth + 1)
        
        if len(data) > 1:
            parse_comments(data[1]['data']['children'])
        
        return {
            'platform': 'reddit',
            'type': 'post',
            'post': post,
            'comments': comments,
            'comment_count': len(comments),
            'scraped_at': time.time(),
            'method': 'json'
        }
    
    def _handle_block(self):
        """Handle being blocked"""
        self.logger.warning("Handling block - rotating IP")
        
        # Clear sessions
        self.sessions.clear()
        
        # Increment rotation counter
        self.stats['ip_rotations'] += 1
        
        # Add delay
        delay = random.uniform(30, 60)
        self.logger.info(f"Waiting {delay:.0f} seconds before retry")
        time.sleep(delay)


class RateLimitException(Exception):
    """Raised when rate limit is exceeded"""
    pass


class BlockedException(Exception):
    """Raised when scraping is blocked"""
    pass