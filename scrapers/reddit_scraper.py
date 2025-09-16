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
import os
import praw
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from utils.reddit_auth_manager import RedditAuthManager
from utils.ip_manager import IPManager, IPType
from utils.rate_limiter import EnhancedRateLimiter
from utils.session_manager import SessionManager
from utils.anti_detection_v2 import AntiDetectionV2

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
        self.session_manager = SessionManager()
        self.anti_v2 = AntiDetectionV2()
        
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
        compliance_mode = os.getenv('COMPLIANCE_MODE', 'api_first').lower()
        
        # Parse URL
        parsed = self._parse_reddit_url(url)
        
        # Determine method order based on compliance mode
        methods = list(self.methods)
        if compliance_mode in ['api_only', 'api_first']:
            methods = ['api', 'json', 'old_reddit']
        elif compliance_mode == 'json_first':
            methods = ['json', 'api', 'old_reddit']
        elif compliance_mode == 'web_first':
            methods = ['old_reddit', 'json', 'api']
        
        # Try each method in order
        for method in methods:
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
        
        # Make request with session manager and AntiDetectionV2 headers
        headers = self.anti_v2.build_headers('reddit', minimal=os.getenv('COMPLIANCE_MODE','').lower() in ['api_only','compliant'])
        proxy = {'url': ip_config.to_proxy_dict()['http']} if ip_config and ip_config.port else None
        session = self.session_manager.get_session('reddit.com', proxy=proxy, headers=headers)
        
        try:
            response = session.get(url, timeout=30, allow_redirects=False)
            
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
                # Pushshift fallback for subreddit and post when .json fails
                if parsed['type'] in ['subreddit', 'post']:
                    ps = self._pushshift_fallback(parsed, params)
                    if ps:
                        return ps
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

    def _pushshift_fallback(self, parsed: Dict, params: Dict) -> Optional[Dict]:
        """Fallback to Pushshift for basic data when Reddit JSON blocks or fails.
        Note: Pushshift availability changes over time; handle gracefully."""
        try:
            base = 'https://api.pushshift.io/reddit'
            if parsed['type'] == 'subreddit':
                q = {
                    'subreddit': parsed['subreddit'],
                    'size': params.get('limit', 25),
                    'sort_type': 'created',
                    'sort': 'desc'
                }
                r = requests.get(f'{base}/search/submission', params=q, timeout=20)
                if r.status_code == 200:
                    data = r.json().get('data', [])
                    posts = []
                    for d in data:
                        posts.append({
                            'id': d.get('id'),
                            'title': d.get('title',''),
                            'author': d.get('author','[unknown]'),
                            'created_utc': d.get('created_utc'),
                            'score': d.get('score',0),
                            'num_comments': d.get('num_comments',0),
                            'permalink': f"https://reddit.com{d.get('permalink','')}",
                            'url': d.get('url'),
                            'selftext': d.get('selftext',''),
                            'subreddit': d.get('subreddit')
                        })
                    return {
                        'platform': 'reddit',
                        'type': 'subreddit',
                        'subreddit': parsed['subreddit'],
                        'posts': posts,
                        'count': len(posts),
                        'scraped_at': time.time(),
                        'method': 'pushshift'
                    }
            elif parsed['type'] == 'post' and parsed.get('post_id'):
                # Fetch submission and comments separately
                r = requests.get(f"{base}/submission/search", params={'ids': parsed['post_id']}, timeout=20)
                c = requests.get(f"{base}/comment/search", params={'link_id': parsed['post_id'], 'size': 2000}, timeout=20)
                if r.status_code == 200:
                    arr = r.json().get('data', [])
                    if arr:
                        pd = arr[0]
                        post = {
                            'id': pd.get('id'),
                            'title': pd.get('title',''),
                            'author': pd.get('author','[unknown]'),
                            'created_utc': pd.get('created_utc'),
                            'score': pd.get('score',0),
                            'num_comments': pd.get('num_comments',0),
                            'selftext': pd.get('selftext',''),
                            'url': pd.get('url'),
                            'subreddit': pd.get('subreddit')
                        }
                        comments = []
                        if c.status_code == 200:
                            for cm in c.json().get('data', []):
                                comments.append({
                                    'id': cm.get('id'),
                                    'author': cm.get('author','[unknown]'),
                                    'body': cm.get('body',''),
                                    'score': cm.get('score',0),
                                    'created_utc': cm.get('created_utc'),
                                    'parent_id': cm.get('parent_id'),
                                    'depth': cm.get('depth',0)
                                })
                        return {
                            'platform': 'reddit',
                            'type': 'post',
                            'post': post,
                            'comments': comments,
                            'comment_count': len(comments),
                            'scraped_at': time.time(),
                            'method': 'pushshift'
                        }
        except Exception:
            return None

    
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