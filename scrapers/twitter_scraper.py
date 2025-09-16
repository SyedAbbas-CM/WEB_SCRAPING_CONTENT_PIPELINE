# scrapers/twitter_scraper.py
"""
Twitter/X Scraper with multiple methods
Supports API v2, GraphQL endpoints, and web scraping
"""

import re
import json
import time
import random
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote, urlencode
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from utils.anti_detection import AntiDetection
from utils.session_manager import SessionManager
from utils.anti_detection_v2 import AntiDetectionV2

class TwitterScraper:
    """
    Advanced Twitter/X scraper supporting:
    - Twitter API v2 (with auth)
    - GraphQL endpoints
    - Selenium web scraping
    - nitter instances (fallback)
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'twitter_scraper.{node_id}')
        
        # Anti-detection
        self.anti_detection = AntiDetection()
        self.anti_v2 = AntiDetectionV2()
        self.session_manager = SessionManager()
        
        # Session management
        self.session = requests.Session()
        self.driver = None
        
        # Twitter API endpoints
        self.api_base = 'https://api.twitter.com/2'
        self.graphql_base = 'https://twitter.com/i/api/graphql'
        
        # Authentication tokens
        self.bearer_token = self._get_bearer_token()
        self.guest_token = None
        
        # Nitter instances for fallback
        self.nitter_instances = [
            'nitter.net',
            'nitter.42l.fr',
            'nitter.pussthecat.org'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0
        
    def _get_bearer_token(self) -> str:
        """Get Twitter bearer token"""
        # Default public bearer token (may change)
        return 'AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
    
    def _ensure_guest_token(self):
        """Ensure we have a valid guest token"""
        if not self.guest_token:
            headers = {
                'Authorization': f'Bearer {self.bearer_token}'
            }
            
            response = self.session.post(
                'https://api.twitter.com/1.1/guest/activate.json',
                headers=headers
            )
            
            if response.status_code == 200:
                self.guest_token = response.json()['guest_token']
                self.logger.info(f"Got guest token: {self.guest_token[:10]}...")
            else:
                raise Exception("Failed to get guest token")
    
    def set_proxy(self, proxy: Dict):
        """Set proxy for requests"""
        if not proxy:
            return
        # Support both single 'url' field and component fields
        proxy_url = proxy.get('url')
        if not proxy_url and all(k in proxy for k in ['address', 'port']):
            auth = ''
            if proxy.get('username') and proxy.get('password'):
                auth = f"{proxy['username']}:{proxy['password']}@"
            proxy_url = f"http://{auth}{proxy['address']}:{proxy['port']}"
        if proxy_url:
            self.session.proxies = {'http': proxy_url, 'https': proxy_url}
    
    def set_headers(self, headers: Dict):
        """Set custom headers"""
        self.session.headers.update(headers)
    
    def scrape_tweet(self, url: str) -> Dict:
        """Scrape a single tweet"""
        # Extract tweet ID from URL
        tweet_id = self._extract_tweet_id(url)
        if not tweet_id:
            raise ValueError(f"Invalid tweet URL: {url}")
        
        # Try methods in order
        methods = [
            self._scrape_tweet_graphql,
            self._scrape_tweet_api,
            self._scrape_tweet_web,
            self._scrape_tweet_nitter
        ]
        
        for method in methods:
            try:
                result = method(tweet_id)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All scraping methods failed")
    
    def scrape_user_timeline(self, username: str, limit: int = 50) -> Dict:
        """Scrape user timeline"""
        # Remove @ if present
        username = username.replace('@', '').split('/')[-1]
        
        # Try methods in order
        methods = [
            self._scrape_timeline_graphql,
            self._scrape_timeline_web,
            self._scrape_timeline_nitter
        ]
        
        for method in methods:
            try:
                result = method(username, limit)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All timeline scraping methods failed")
    
    def scrape_search(self, query: str, params: Dict) -> Dict:
        """Scrape search results"""
        search_type = params.get('type', 'latest')  # latest, top, people, photos, videos
        limit = params.get('limit', 50)
        
        # Try methods in order
        methods = [
            self._search_graphql,
            self._search_web,
            self._search_nitter
        ]
        
        for method in methods:
            try:
                result = method(query, search_type, limit)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All search methods failed")
    
    def _extract_tweet_id(self, url: str) -> Optional[str]:
        """Extract tweet ID from URL"""
        patterns = [
            r'twitter\.com/\w+/status/(\d+)',
            r'x\.com/\w+/status/(\d+)',
            r'mobile\.twitter\.com/\w+/status/(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Check if it's already just an ID
        if url.isdigit():
            return url
        
        return None
    
    def _scrape_tweet_graphql(self, tweet_id: str) -> Dict:
        """Scrape tweet using GraphQL endpoint"""
        self._ensure_guest_token()
        
        # GraphQL query ID for TweetDetail
        query_id = 'VWFGPVAGkZMGRKGe3GFFnA'
        
        variables = {
            "focalTweetId": tweet_id,
            "with_rux_injections": False,
            "includePromotedContent": True,
            "withCommunity": True,
            "withQuickPromoteEligibilityTweetFields": True,
            "withBirdwatchNotes": True,
            "withVoice": True,
            "withV2Timeline": True
        }
        
        features = {
            "rweb_lists_timeline_redesign_enabled": True,
            "responsive_web_graphql_exclude_directive_enabled": True,
            "verified_phone_label_enabled": False,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "tweetypie_unmention_optimization_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "responsive_web_twitter_article_tweet_consumption_enabled": False,
            "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
            "longform_notetweets_rich_text_read_enabled": True,
            "longform_notetweets_inline_media_enabled": True,
            "responsive_web_enhance_cards_enabled": False
        }
        
        params = {
            'variables': json.dumps(variables),
            'features': json.dumps(features)
        }
        
        headers = self.anti_v2.build_headers('twitter', minimal=os.getenv('COMPLIANCE_MODE','').lower() in ['api_only','compliant'])
        headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'X-Guest-Token': self.guest_token,
            'Content-Type': 'application/json'
        })
        
        url = f'{self.graphql_base}/{query_id}/TweetDetail'
        response = self.session.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_graphql_tweet(data)
        else:
            raise Exception(f"GraphQL request failed: {response.status_code}")
    
    def _scrape_tweet_api(self, tweet_id: str) -> Dict:
        """Scrape tweet using API v2"""
        self._ensure_guest_token()
        
        url = f'{self.api_base}/tweets/{tweet_id}'
        
        params = {
            'expansions': 'author_id,referenced_tweets.id,attachments.media_keys',
            'tweet.fields': 'created_at,author_id,conversation_id,public_metrics,lang,context_annotations,entities,referenced_tweets',
            'user.fields': 'name,username,verified,profile_image_url',
            'media.fields': 'url,preview_image_url,type,duration_ms'
        }
        
        headers = self.anti_v2.build_headers('twitter', minimal=os.getenv('COMPLIANCE_MODE','').lower() in ['api_only','compliant'])
        headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'X-Guest-Token': self.guest_token
        })
        
        response = self.session.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_api_tweet(data)
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def _scrape_tweet_web(self, tweet_id: str) -> Dict:
        """Scrape tweet using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f'https://twitter.com/i/status/{tweet_id}'
        self.driver.get(url)
        
        # Wait for tweet to load
        try:
            tweet_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//article[@data-testid="tweet"]'))
            )
        except TimeoutException:
            raise Exception("Tweet not found or timeout")
        
        # Extract tweet data
        tweet_data = {}
        
        # Text content
        try:
            text_element = tweet_element.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            tweet_data['text'] = text_element.text
        except NoSuchElementException:
            tweet_data['text'] = ''
        
        # Author
        try:
            author_element = tweet_element.find_element(By.XPATH, './/div[@data-testid="User-Names"]')
            username = author_element.find_element(By.XPATH, './/span[contains(text(), "@")]').text
            display_name = author_element.find_element(By.XPATH, './/span[not(contains(text(), "@"))]').text
            
            tweet_data['author'] = {
                'username': username.replace('@', ''),
                'display_name': display_name
            }
        except NoSuchElementException:
            tweet_data['author'] = {}
        
        # Metrics
        metrics = {}
        metric_types = ['reply', 'retweet', 'like']
        
        for metric_type in metric_types:
            try:
                metric_element = tweet_element.find_element(
                    By.XPATH, f'.//div[@data-testid="{metric_type}"]//span'
                )
                count_text = metric_element.text
                metrics[f'{metric_type}_count'] = self._parse_metric_count(count_text)
            except NoSuchElementException:
                metrics[f'{metric_type}_count'] = 0
        
        tweet_data['metrics'] = metrics
        
        # Time
        try:
            time_element = tweet_element.find_element(By.XPATH, './/time')
            tweet_data['created_at'] = time_element.get_attribute('datetime')
        except NoSuchElementException:
            tweet_data['created_at'] = None
        
        # Media
        media = []
        try:
            img_elements = tweet_element.find_elements(By.XPATH, './/img[contains(@src, "media")]')
            for img in img_elements:
                media.append({
                    'type': 'photo',
                    'url': img.get_attribute('src')
                })
        except:
            pass
        
        tweet_data['media'] = media
        
        return {
            'platform': 'twitter',
            'type': 'tweet',
            'id': tweet_id,
            'url': url,
            'data': tweet_data,
            'scraped_at': time.time(),
            'method': 'web'
        }
    
    def _scrape_tweet_nitter(self, tweet_id: str) -> Dict:
        """Scrape tweet using Nitter instance"""
        for instance in self.nitter_instances:
            try:
                url = f'https://{instance}/i/status/{tweet_id}'
                headers = self.anti_detection.get_headers('generic')
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    # Parse Nitter HTML
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract tweet data
                    tweet_data = {}
                    
                    # Main tweet content
                    main_tweet = soup.find('div', class_='main-tweet')
                    if main_tweet:
                        # Text
                        content = main_tweet.find('div', class_='tweet-content')
                        if content:
                            tweet_data['text'] = content.get_text(strip=True)
                        
                        # Author
                        author_elem = main_tweet.find('a', class_='username')
                        if author_elem:
                            tweet_data['author'] = {
                                'username': author_elem.text.replace('@', ''),
                                'display_name': main_tweet.find('a', class_='fullname').text
                            }
                        
                        # Stats
                        stats = main_tweet.find('div', class_='tweet-stats')
                        if stats:
                            metrics = {}
                            for stat in stats.find_all('span', class_='tweet-stat'):
                                icon = stat.find('div', class_='icon-container')
                                if icon:
                                    value = stat.find('span').text.strip()
                                    if 'comment' in str(icon):
                                        metrics['reply_count'] = self._parse_metric_count(value)
                                    elif 'retweet' in str(icon):
                                        metrics['retweet_count'] = self._parse_metric_count(value)
                                    elif 'heart' in str(icon):
                                        metrics['like_count'] = self._parse_metric_count(value)
                            
                            tweet_data['metrics'] = metrics
                        
                        # Time
                        date_elem = main_tweet.find('span', class_='tweet-date')
                        if date_elem:
                            tweet_data['created_at'] = date_elem.find('a')['title']
                    
                    return {
                        'platform': 'twitter',
                        'type': 'tweet',
                        'id': tweet_id,
                        'url': f'https://twitter.com/i/status/{tweet_id}',
                        'data': tweet_data,
                        'scraped_at': time.time(),
                        'method': 'nitter'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Nitter instance {instance} failed: {e}")
                continue
        
        raise Exception("All Nitter instances failed")
    
    def _scrape_timeline_graphql(self, username: str, limit: int) -> Dict:
        """Scrape user timeline using GraphQL"""
        self._ensure_guest_token()
        
        # First, get user ID
        user_id = self._get_user_id(username)
        if not user_id:
            raise Exception(f"User not found: {username}")
        
        # GraphQL query for UserTweets
        query_id = 'XicnWRbyQ3WgVY__VataBQ'
        
        variables = {
            "userId": user_id,
            "count": min(limit, 100),
            "includePromotedContent": True,
            "withQuickPromoteEligibilityTweetFields": True,
            "withVoice": True,
            "withV2Timeline": True
        }
        
        headers = self.anti_detection.get_headers('twitter')
        headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'X-Guest-Token': self.guest_token
        })
        
        tweets = []
        cursor = None
        
        while len(tweets) < limit:
            if cursor:
                variables['cursor'] = cursor
            
            params = {
                'variables': json.dumps(variables),
                'features': json.dumps(self._get_default_features())
            }
            
            url = f'{self.graphql_base}/{query_id}/UserTweets'
            response = self.session.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                entries = self._extract_timeline_entries(data)
                
                for entry in entries:
                    tweet = self._parse_timeline_tweet(entry)
                    if tweet:
                        tweets.append(tweet)
                
                # Get next cursor
                cursor = self._extract_cursor(data)
                if not cursor:
                    break
            else:
                break
        
        return {
            'platform': 'twitter',
            'type': 'timeline',
            'username': username,
            'tweets': tweets[:limit],
            'count': len(tweets),
            'scraped_at': time.time(),
            'method': 'graphql'
        }
    
    def _scrape_timeline_web(self, username: str, limit: int) -> Dict:
        """Scrape user timeline using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f'https://twitter.com/{username}'
        self.driver.get(url)
        
        tweets = []
        seen_ids = set()
        
        # Scroll and collect tweets
        last_height = 0
        no_change_count = 0
        
        while len(tweets) < limit and no_change_count < 3:
            # Get all tweet elements
            tweet_elements = self.driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
            
            for element in tweet_elements:
                try:
                    # Extract tweet ID
                    link_elem = element.find_element(By.XPATH, './/a[contains(@href, "/status/")]')
                    tweet_url = link_elem.get_attribute('href')
                    tweet_id = tweet_url.split('/status/')[-1].split('?')[0]
                    
                    if tweet_id not in seen_ids:
                        seen_ids.add(tweet_id)
                        
                        # Extract basic tweet data
                        tweet_data = self._extract_tweet_from_element(element)
                        tweet_data['id'] = tweet_id
                        tweets.append(tweet_data)
                        
                except Exception as e:
                    continue
            
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Check if we've reached the bottom
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                no_change_count += 1
            else:
                no_change_count = 0
            last_height = new_height
        
        return {
            'platform': 'twitter',
            'type': 'timeline',
            'username': username,
            'tweets': tweets[:limit],
            'count': len(tweets),
            'scraped_at': time.time(),
            'method': 'web'
        }
    
    def _scrape_timeline_nitter(self, username: str, limit: int) -> Dict:
        """Scrape user timeline using Nitter"""
        for instance in self.nitter_instances:
            try:
                url = f'https://{instance}/{username}'
                headers = self.anti_detection.get_headers('generic')
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    tweets = []
                    timeline = soup.find('div', class_='timeline')
                    
                    if timeline:
                        for item in timeline.find_all('div', class_='timeline-item'):
                            if len(tweets) >= limit:
                                break
                            
                            tweet = self._parse_nitter_tweet(item)
                            if tweet:
                                tweets.append(tweet)
                    
                    return {
                        'platform': 'twitter',
                        'type': 'timeline',
                        'username': username,
                        'tweets': tweets,
                        'count': len(tweets),
                        'scraped_at': time.time(),
                        'method': 'nitter'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Nitter timeline failed for {instance}: {e}")
                continue
        
        raise Exception("All Nitter instances failed")
    
    def _search_graphql(self, query: str, search_type: str, limit: int) -> Dict:
        """Search using GraphQL"""
        self._ensure_guest_token()
        
        # GraphQL query for SearchTimeline
        query_id = 'gkjsKepM6gl_HmFWoWKfgg'
        
        variables = {
            "rawQuery": query,
            "count": min(limit, 100),
            "querySource": "typed_query",
            "product": search_type  # Latest, Top, People, Photos, Videos
        }
        
        headers = self.anti_detection.get_headers('twitter')
        headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'X-Guest-Token': self.guest_token
        })
        
        results = []
        cursor = None
        
        while len(results) < limit:
            if cursor:
                variables['cursor'] = cursor
            
            params = {
                'variables': json.dumps(variables),
                'features': json.dumps(self._get_default_features())
            }
            
            url = f'{self.graphql_base}/{query_id}/SearchTimeline'
            response = self.session.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                entries = self._extract_timeline_entries(data)
                
                for entry in entries:
                    if search_type == 'People':
                        user = self._parse_user_result(entry)
                        if user:
                            results.append(user)
                    else:
                        tweet = self._parse_timeline_tweet(entry)
                        if tweet:
                            results.append(tweet)
                
                # Get next cursor
                cursor = self._extract_cursor(data)
                if not cursor:
                    break
            else:
                break
        
        return {
            'platform': 'twitter',
            'type': 'search',
            'query': query,
            'search_type': search_type,
            'results': results[:limit],
            'count': len(results),
            'scraped_at': time.time(),
            'method': 'graphql'
        }
    
    def _init_selenium_driver(self):
        """Initialize Selenium driver with anti-detection"""
        options = webdriver.ChromeOptions()
        
        # Anti-detection options
        selenium_options = self.anti_detection.get_selenium_options()
        for arg in selenium_options['arguments']:
            options.add_argument(arg)
        
        for key, value in selenium_options['prefs'].items():
            options.add_experimental_option('prefs', {key: value})
        
        options.add_experimental_option('excludeSwitches', selenium_options['excludeSwitches'])
        options.add_experimental_option('useAutomationExtension', selenium_options['useAutomationExtension'])
        
        # Create driver
        self.driver = webdriver.Chrome(options=options)
        
        # Inject stealth JavaScript
        stealth_js = self.anti_detection.get_stealth_js_injection()
        self.driver.execute_script(stealth_js)
    
    def _get_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username"""
        # Use UserByScreenName GraphQL query
        query_id = 'G3KGOASz96M-Qu0nwmGXNg'
        
        variables = {
            "screen_name": username,
            "withSafetyModeUserFields": True
        }
        
        params = {
            'variables': json.dumps(variables),
            'features': json.dumps(self._get_default_features())
        }
        
        headers = self.anti_detection.get_headers('twitter')
        headers.update({
            'Authorization': f'Bearer {self.bearer_token}',
            'X-Guest-Token': self.guest_token
        })
        
        url = f'{self.graphql_base}/{query_id}/UserByScreenName'
        response = self.session.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            try:
                user = data['data']['user']['result']
                return user['rest_id']
            except:
                return None
        
        return None
    
    def _get_default_features(self) -> Dict:
        """Get default GraphQL features"""
        return {
            "rweb_lists_timeline_redesign_enabled": True,
            "responsive_web_graphql_exclude_directive_enabled": True,
            "verified_phone_label_enabled": False,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "tweetypie_unmention_optimization_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "tweet_awards_web_tipping_enabled": False,
            "freedom_of_speech_not_reach_fetch_enabled": True,
            "standardized_nudges_misinfo": True,
            "longform_notetweets_rich_text_read_enabled": True,
            "responsive_web_enhance_cards_enabled": False
        }
    
    def _parse_metric_count(self, text: str) -> int:
        """Parse metric count from text (1.2K -> 1200)"""
        if not text:
            return 0
        
        text = text.strip().upper()
        
        if 'K' in text:
            return int(float(text.replace('K', '')) * 1000)
        elif 'M' in text:
            return int(float(text.replace('M', '')) * 1000000)
        else:
            return int(text) if text.isdigit() else 0
    
    def _extract_timeline_entries(self, data: Dict) -> List[Dict]:
        """Extract entries from timeline response"""
        entries = []
        
        try:
            instructions = data['data']['user']['result']['timeline_v2']['timeline']['instructions']
            
            for instruction in instructions:
                if instruction['type'] == 'TimelineAddEntries':
                    entries.extend(instruction['entries'])
                elif instruction['type'] == 'TimelineReplaceEntry':
                    entries.append(instruction['entry'])
        except:
            pass
        
        return entries
    
    def _extract_cursor(self, data: Dict) -> Optional[str]:
        """Extract pagination cursor"""
        try:
            instructions = data['data']['user']['result']['timeline_v2']['timeline']['instructions']
            
            for instruction in instructions:
                if instruction['type'] == 'TimelineAddEntries':
                    for entry in instruction['entries']:
                        if entry['entryId'].startswith('cursor-bottom'):
                            return entry['content']['value']
        except:
            pass
        
        return None
    
    def _parse_graphql_tweet(self, data: Dict) -> Dict:
        """Parse tweet from GraphQL response"""
        try:
            # Navigate through the response structure
            entries = data['data']['threaded_conversation_with_injections_v2']['instructions'][0]['entries']
            
            for entry in entries:
                if 'tweet' in entry['entryId']:
                    tweet_result = entry['content']['itemContent']['tweet_results']['result']
                    
                    # Extract tweet data
                    legacy = tweet_result['legacy']
                    
                    return {
                        'id': tweet_result['rest_id'],
                        'text': legacy['full_text'],
                        'created_at': legacy['created_at'],
                        'author': {
                            'username': tweet_result['core']['user_results']['result']['legacy']['screen_name'],
                            'display_name': tweet_result['core']['user_results']['result']['legacy']['name']
                        },
                        'metrics': {
                            'reply_count': legacy['reply_count'],
                            'retweet_count': legacy['retweet_count'],
                            'like_count': legacy['favorite_count'],
                            'view_count': tweet_result.get('views', {}).get('count', 0)
                        },
                        'lang': legacy['lang'],
                        'source': legacy['source']
                    }
        except Exception as e:
            self.logger.error(f"Failed to parse GraphQL tweet: {e}")
            return {}
    
    def _parse_api_tweet(self, data: Dict) -> Dict:
        """Parse tweet from API v2 response"""
        try:
            tweet = data['data']
            includes = data.get('includes', {})
            
            # Get author info
            author = {}
            if 'users' in includes:
                for user in includes['users']:
                    if user['id'] == tweet['author_id']:
                        author = {
                            'username': user['username'],
                            'display_name': user['name'],
                            'verified': user.get('verified', False)
                        }
                        break
            
            return {
                'id': tweet['id'],
                'text': tweet['text'],
                'created_at': tweet['created_at'],
                'author': author,
                'metrics': tweet.get('public_metrics', {}),
                'lang': tweet.get('lang'),
                'conversation_id': tweet.get('conversation_id')
            }
        except Exception as e:
            self.logger.error(f"Failed to parse API tweet: {e}")
            return {}
    
    def _parse_timeline_tweet(self, entry: Dict) -> Optional[Dict]:
        """Parse tweet from timeline entry"""
        try:
            if 'itemContent' not in entry['content']:
                return None
            
            tweet_result = entry['content']['itemContent']['tweet_results']['result']
            
            # Skip if it's a promoted tweet
            if 'promotedMetadata' in entry['content']['itemContent']:
                return None
            
            legacy = tweet_result['legacy']
            user = tweet_result['core']['user_results']['result']['legacy']
            
            return {
                'id': tweet_result['rest_id'],
                'text': legacy['full_text'],
                'created_at': legacy['created_at'],
                'author': {
                    'username': user['screen_name'],
                    'display_name': user['name']
                },
                'metrics': {
                    'reply_count': legacy['reply_count'],
                    'retweet_count': legacy['retweet_count'],
                    'like_count': legacy['favorite_count']
                }
            }
        except:
            return None
    
    def _extract_tweet_from_element(self, element) -> Dict:
        """Extract tweet data from Selenium element"""
        tweet_data = {}
        
        try:
            # Text
            text_elem = element.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            tweet_data['text'] = text_elem.text
        except:
            tweet_data['text'] = ''
        
        try:
            # Time
            time_elem = element.find_element(By.XPATH, './/time')
            tweet_data['created_at'] = time_elem.get_attribute('datetime')
        except:
            tweet_data['created_at'] = None
        
        # Metrics
        metrics = {}
        for metric in ['reply', 'retweet', 'like']:
            try:
                elem = element.find_element(By.XPATH, f'.//div[@data-testid="{metric}"]//span')
                metrics[f'{metric}_count'] = self._parse_metric_count(elem.text)
            except:
                metrics[f'{metric}_count'] = 0
        
        tweet_data['metrics'] = metrics
        
        return tweet_data
    
    def _parse_nitter_tweet(self, item) -> Optional[Dict]:
        """Parse tweet from Nitter timeline item"""
        try:
            tweet_data = {}
            
            # Tweet link and ID
            link = item.find('a', class_='tweet-link')
            if link:
                tweet_id = link['href'].split('/')[-1].replace('#m', '')
                tweet_data['id'] = tweet_id
            
            # Content
            content = item.find('div', class_='tweet-content')
            if content:
                tweet_data['text'] = content.get_text(strip=True)
            
            # Time
            date = item.find('span', class_='tweet-date')
            if date:
                tweet_data['created_at'] = date.find('a')['title']
            
            # Stats
            stats = item.find('div', class_='tweet-stats')
            if stats:
                metrics = {}
                for icon in stats.find_all('div', class_='icon-container'):
                    count = icon.find_next_sibling('span')
                    if count:
                        value = self._parse_metric_count(count.text)
                        if 'comment' in str(icon):
                            metrics['reply_count'] = value
                        elif 'retweet' in str(icon):
                            metrics['retweet_count'] = value
                        elif 'heart' in str(icon):
                            metrics['like_count'] = value
                
                tweet_data['metrics'] = metrics
            
            return tweet_data
            
        except:
            return None
    
    def _parse_user_result(self, entry: Dict) -> Optional[Dict]:
        """Parse user from search results"""
        try:
            user_result = entry['content']['itemContent']['user_results']['result']
            legacy = user_result['legacy']
            
            return {
                'id': user_result['rest_id'],
                'username': legacy['screen_name'],
                'display_name': legacy['name'],
                'bio': legacy['description'],
                'verified': legacy.get('verified', False),
                'followers_count': legacy['followers_count'],
                'following_count': legacy['friends_count'],
                'tweet_count': legacy['statuses_count'],
                'created_at': legacy['created_at']
            }
        except:
            return None
    
    def __del__(self):
        """Cleanup"""
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass