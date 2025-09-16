            'likers': 'd5d763b1e2acf209d62d1ce78df0e665'
        }
        
        # Alternative services
        self.alt_services = [
            'picuki.com',
            'imginn.com',
            'pixwox.com'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 3.0
    
    def login(self, username: str, password: str) -> bool:
        """Login to Instagram"""
        try:
            # Get initial cookies
            self._get_initial_cookies()
            
            # Prepare login data
            enc_password = f'#PWD_INSTAGRAM_BROWSER:0:{int(time.time())}:{password}'
            
            login_data = {
                'username': username,
                'enc_password': enc_password,
                'queryParams': {},
                'optIntoOneTap': False
            }
            
            headers = self.anti_detection.get_headers('instagram')
            headers.update({
                'X-CSRFToken': self.csrf_token,
                'X-IG-App-ID': '936619743392459',
                'X-Instagram-AJAX': '1006598911',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f'{self.base_url}/'
            })
            
            # Send login request
            response = self.session.post(
                f'{self.base_url}/accounts/login/ajax/',
                data=login_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('authenticated'):
                    self.session_id = self.session.cookies.get('sessionid')
                    self.logger.info(f"Successfully logged in as {username}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            return False
    
    def _get_initial_cookies(self):
        """Get initial cookies and tokens"""
        headers = self.anti_detection.get_headers('instagram')
        response = self.session.get(self.base_url, headers=headers)
        
        # Extract CSRF token
        match = re.search(r'"csrf_token":"([^"]+)"', response.text)
        if match:
            self.csrf_token = match.group(1)
        
        # Set user agent
        self.user_agent = headers['User-Agent']
    
    def set_proxy(self, proxy: Dict):
        """Set proxy for requests"""
        if proxy:
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['address']}:{proxy['port']}"
            self.session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
    
    def scrape_post(self, url: str) -> Dict:
        """Scrape a single post"""
        # Extract shortcode from URL
        shortcode = self._extract_shortcode(url)
        if not shortcode:
            raise ValueError(f"Invalid post URL: {url}")
        
        # Try methods in order
        methods = [
            self._scrape_post_graphql,
            self._scrape_post_api,
            self._scrape_post_web,
            self._scrape_post_alt
        ]
        
        for method in methods:
            try:
                result = method(shortcode)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All scraping methods failed")
    
    def scrape_user_posts(self, username: str, limit: int = 50) -> Dict:
        """Scrape user posts"""
        username = username.replace('@', '').split('/')[-1]
        
        # Try methods in order
        methods = [
            self._scrape_user_posts_graphql,
            self._scrape_user_posts_api,
            self._scrape_user_posts_web,
            self._scrape_user_posts_alt
        ]
        
        for method in methods:
            try:
                result = method(username, limit)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All user posts scraping methods failed")
    
    def scrape_hashtag(self, hashtag: str, limit: int = 50) -> Dict:
        """Scrape hashtag posts"""
        hashtag = hashtag.replace('#', '')
        
        methods = [
            self._scrape_hashtag_graphql,
            self._scrape_hashtag_web,
            self._scrape_hashtag_alt
        ]
        
        for method in methods:
            try:
                result = method(hashtag, limit)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All hashtag scraping methods failed")
    
    def scrape_stories(self, username: str) -> Dict:
        """Scrape user stories"""
        username = username.replace('@', '').split('/')[-1]
        
        methods = [
            self._scrape_stories_api,
            self._scrape_stories_web
        ]
        
        for method in methods:
            try:
                result = method(username)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All stories scraping methods failed")
    
    def _extract_shortcode(self, url: str) -> Optional[str]:
        """Extract shortcode from URL"""
        patterns = [
            r'instagram\.com/p/([A-Za-z0-9_-]+)',
            r'instagram\.com/reel/([A-Za-z0-9_-]+)',
            r'instagram\.com/tv/([A-Za-z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Check if it's already a shortcode
        if re.match(r'^[A-Za-z0-9_-]+', url):
            return url
        
        return None

    # --- File corruption below was removed and restored to a single coherent implementation ---

import re
import json
import time
import random
import hashlib
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote, urlencode
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from utils.anti_detection import AntiDetection

class InstagramScraper:
    """
    Instagram scraper with multiple methods:
    - GraphQL API (requires session)
    - Web scraping with Selenium
    - Mobile API endpoints
    - Picuki/Imginn fallback
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'instagram_scraper.{node_id}')
        
        # Anti-detection
        self.anti_detection = AntiDetection()
        
        # Session management
        self.session = requests.Session()
        self.driver = None
        
        # Instagram API endpoints
        self.base_url = 'https://www.instagram.com'
        self.api_url = 'https://i.instagram.com/api/v1'
        self.graphql_url = f'{self.base_url}/graphql/query'
        
        # Authentication
        self.session_id = None
        self.csrf_token = None
        self.user_agent = None
        
        # Query hashes for GraphQL
        self.query_hashes = {
            'user_posts': 'e769aa130647d2354c40ea37b90b7c8',
            'post_details': '2b0673e0dc4580674a88d426fe00ea90',
            'user_info': 'd4d88dc1500312af6f937f7b804c68c3',
            'hashtag_posts': 'f92f56d47dc7a55b606908374b43a314',
            'location_posts': '1b84447a4d8b6d6d0426fefb34514485',
            'stories': 'de8017ee0a7c9c45ec4260733d81ea31',
            'highlights': 'd4d88dc1500312af6f937f7b804c68c3',
            'comments': 'bc3296d1ce80a24b1b6e40b1e72903f5',
            'likers': 'd5d763b1e2acf209d62, url):
            return url
        
        return None
    
    def _scrape_post_graphql(self, shortcode: str) -> Dict:
        """Scrape post using GraphQL"""
        self._rate_limit()
        
        variables = {
            'shortcode': shortcode,
            'child_comment_count': 3,
            'fetch_comment_count': 40,
            'parent_comment_count': 24,
            'has_threaded_comments': True
        }
        
        params = {
            'query_hash': self.query_hashes['post_details'],
            'variables': json.dumps(variables)
        }
        
        headers = self.anti_detection.get_headers('instagram')
        headers.update({
            'X-IG-App-ID': '936619743392459',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': f'{self.base_url}/p/{shortcode}/'
        })
        
        response = self.session.get(self.graphql_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_graphql_post(data['data']['shortcode_media'])
        
        raise Exception(f"GraphQL request failed: {response.status_code}")
    
    def _scrape_post_api(self, shortcode: str) -> Dict:
        """Scrape post using mobile API"""
        self._rate_limit()
        
        # Convert shortcode to media ID
        media_id = self._shortcode_to_media_id(shortcode)
        
        headers = self.anti_detection.get_headers('instagram')
        headers.update({
            'User-Agent': 'Instagram 121.0.0.29.119 Android',
            'X-IG-Capabilities': '3brTvw=='
        })
        
        url = f'{self.api_url}/media/{media_id}/info/'
        response = self.session.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_api_post(data['items'][0])
        
        raise Exception(f"API request failed: {response.status_code}")
    
    def _scrape_post_web(self, shortcode: str) -> Dict:
        """Scrape post using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f'{self.base_url}/p/{shortcode}/'
        self.driver.get(url)
        
        # Wait for post to load
        try:
            post_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'article'))
            )
        except TimeoutException:
            raise Exception("Post not found or timeout")
        
        # Extract post data
        post_data = {}
        
        # Image/Video
        media = []
        try:
            # Check for multiple images
            carousel_items = self.driver.find_elements(By.CSS_SELECTOR, 'li[class*="carousel"]')
            if carousel_items:
                for item in carousel_items:
                    img = item.find_element(By.TAG_NAME, 'img')
                    media.append({
                        'type': 'image',
                        'url': img.get_attribute('src'),
                        'alt': img.get_attribute('alt')
                    })
            else:
                # Single image/video
                img = post_element.find_element(By.CSS_SELECTOR, 'img[class*="Photo"]')
                media.append({
                    'type': 'image',
                    'url': img.get_attribute('src'),
                    'alt': img.get_attribute('alt')
                })
        except:
            # Try video
            try:
                video = post_element.find_element(By.TAG_NAME, 'video')
                media.append({
                    'type': 'video',
                    'url': video.get_attribute('src'),
                    'poster': video.get_attribute('poster')
                })
            except:
                pass
        
        post_data['media'] = media
        
        # Caption
        try:
            caption_element = post_element.find_element(By.CSS_SELECTOR, 'div[class*="Caption"] span')
            post_data['caption'] = caption_element.text
        except:
            post_data['caption'] = ''
        
        # Author
        try:
            author_element = post_element.find_element(By.CSS_SELECTOR, 'header a[class*="Username"]')
            post_data['author'] = {
                'username': author_element.text.replace('@', '')
            }
        except:
            post_data['author'] = {}
        
        # Metrics
        metrics = {}
        
        # Likes
        try:
            likes_element = self.driver.find_element(By.XPATH, '//button[contains(@class, "like")]//span')
            metrics['like_count'] = self._parse_count(likes_element.text)
        except:
            metrics['like_count'] = 0
        
        # Comments
        try:
            comments_element = self.driver.find_element(By.XPATH, '//a[contains(@href, "/comments/")]//span')
            metrics['comment_count'] = self._parse_count(comments_element.text)
        except:
            metrics['comment_count'] = 0
        
        post_data['metrics'] = metrics
        
        # Time
        try:
            time_element = post_element.find_element(By.TAG_NAME, 'time')
            post_data['created_at'] = time_element.get_attribute('datetime')
        except:
            post_data['created_at'] = None
        
        return {
            'platform': 'instagram',
            'type': 'post',
            'shortcode': shortcode,
            'url': url,
            'data': post_data,
            'scraped_at': time.time(),
            'method': 'web'
        }
    
    def _scrape_post_alt(self, shortcode: str) -> Dict:
        """Scrape post using alternative services"""
        for service in self.alt_services:
            try:
                if service == 'picuki.com':
                    return self._scrape_picuki_post(shortcode)
                elif service == 'imginn.com':
                    return self._scrape_imginn_post(shortcode)
                elif service == 'pixwox.com':
                    return self._scrape_pixwox_post(shortcode)
            except Exception as e:
                self.logger.warning(f"{service} failed: {e}")
                continue
        
        raise Exception("All alternative services failed")
    
    def _scrape_user_posts_graphql(self, username: str, limit: int) -> Dict:
        """Scrape user posts using GraphQL"""
        # First get user ID
        user_id = self._get_user_id(username)
        if not user_id:
            raise Exception(f"User not found: {username}")
        
        posts = []
        end_cursor = None
        
        while len(posts) < limit:
            variables = {
                'id': user_id,
                'first': min(50, limit - len(posts))
            }
            
            if end_cursor:
                variables['after'] = end_cursor
            
            params = {
                'query_hash': self.query_hashes['user_posts'],
                'variables': json.dumps(variables)
            }
            
            headers = self.anti_detection.get_headers('instagram')
            headers.update({
                'X-IG-App-ID': '936619743392459',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f'{self.base_url}/{username}/'
            })
            
            response = self.session.get(self.graphql_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                timeline = data['data']['user']['edge_owner_to_timeline_media']
                
                for edge in timeline['edges']:
                    post = self._parse_timeline_post(edge['node'])
                    posts.append(post)
                
                # Check for more pages
                page_info = timeline['page_info']
                if page_info['has_next_page']:
                    end_cursor = page_info['end_cursor']
                else:
                    break
            else:
                break
            
            self._rate_limit()
        
        return {
            'platform': 'instagram',
            'type': 'user_posts',
            'username': username,
            'posts': posts[:limit],
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'graphql'
        }
    
    def _scrape_user_posts_api(self, username: str, limit: int) -> Dict:
        """Scrape user posts using mobile API"""
        # Get user ID first
        user_id = self._get_user_id_api(username)
        if not user_id:
            raise Exception(f"User not found: {username}")
        
        posts = []
        max_id = None
        
        while len(posts) < limit:
            url = f'{self.api_url}/users/{user_id}/feed/'
            
            params = {}
            if max_id:
                params['max_id'] = max_id
            
            headers = self.anti_detection.get_headers('instagram')
            headers.update({
                'User-Agent': 'Instagram 121.0.0.29.119 Android'
            })
            
            response = self.session.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    post = self._parse_api_post(item)
                    posts.append(post)
                
                # Check for more pages
                if data.get('more_available') and 'next_max_id' in data:
                    max_id = data['next_max_id']
                else:
                    break
            else:
                break
            
            self._rate_limit()
        
        return {
            'platform': 'instagram',
            'type': 'user_posts',
            'username': username,
            'posts': posts[:limit],
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'api'
        }
    
    def _scrape_user_posts_web(self, username: str, limit: int) -> Dict:
        """Scrape user posts using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f'{self.base_url}/{username}/'
        self.driver.get(url)
        
        # Wait for posts to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'article a[href*="/p/"]'))
            )
        except TimeoutException:
            raise Exception("No posts found or timeout")
        
        posts = []
        seen_shortcodes = set()
        last_height = 0
        no_change_count = 0
        
        while len(posts) < limit and no_change_count < 3:
            # Get all post links
            post_links = self.driver.find_elements(By.CSS_SELECTOR, 'article a[href*="/p/"]')
            
            for link in post_links:
                href = link.get_attribute('href')
                shortcode = self._extract_shortcode(href)
                
                if shortcode and shortcode not in seen_shortcodes:
                    seen_shortcodes.add(shortcode)
                    
                    # Extract basic post data
                    post_data = {
                        'shortcode': shortcode,
                        'url': href
                    }
                    
                    # Try to get image
                    try:
                        img = link.find_element(By.TAG_NAME, 'img')
                        post_data['thumbnail'] = img.get_attribute('src')
                        post_data['alt'] = img.get_attribute('alt')
                    except:
                        pass
                    
                    posts.append(post_data)
            
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
            'platform': 'instagram',
            'type': 'user_posts',
            'username': username,
            'posts': posts[:limit],
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'web'
        }
    
    def _scrape_user_posts_alt(self, username: str, limit: int) -> Dict:
        """Scrape user posts using alternative services"""
        for service in self.alt_services:
            try:
                if service == 'picuki.com':
                    return self._scrape_picuki_user(username, limit)
                elif service == 'imginn.com':
                    return self._scrape_imginn_user(username, limit)
                elif service == 'pixwox.com':
                    return self._scrape_pixwox_user(username, limit)
            except Exception as e:
                self.logger.warning(f"{service} failed: {e}")
                continue
        
        raise Exception("All alternative services failed")
    
    def _scrape_hashtag_graphql(self, hashtag: str, limit: int) -> Dict:
        """Scrape hashtag posts using GraphQL"""
        posts = []
        end_cursor = None
        
        while len(posts) < limit:
            variables = {
                'tag_name': hashtag,
                'first': min(50, limit - len(posts))
            }
            
            if end_cursor:
                variables['after'] = end_cursor
            
            params = {
                'query_hash': self.query_hashes['hashtag_posts'],
                'variables': json.dumps(variables)
            }
            
            headers = self.anti_detection.get_headers('instagram')
            headers.update({
                'X-IG-App-ID': '936619743392459',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f'{self.base_url}/explore/tags/{hashtag}/'
            })
            
            response = self.session.get(self.graphql_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                hashtag_data = data['data']['hashtag']
                
                # Recent posts
                for edge in hashtag_data['edge_hashtag_to_media']['edges']:
                    post = self._parse_timeline_post(edge['node'])
                    posts.append(post)
                
                # Check for more pages
                page_info = hashtag_data['edge_hashtag_to_media']['page_info']
                if page_info['has_next_page']:
                    end_cursor = page_info['end_cursor']
                else:
                    break
            else:
                break
            
            self._rate_limit()
        
        return {
            'platform': 'instagram',
            'type': 'hashtag',
            'hashtag': hashtag,
            'posts': posts[:limit],
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'graphql'
        }
    
    def _scrape_stories_api(self, username: str) -> Dict:
        """Scrape stories using API"""
        # Get user ID
        user_id = self._get_user_id_api(username)
        if not user_id:
            raise Exception(f"User not found: {username}")
        
        url = f'{self.api_url}/feed/user/{user_id}/story/'
        
        headers = self.anti_detection.get_headers('instagram')
        headers.update({
            'User-Agent': 'Instagram 121.0.0.29.119 Android'
        })
        
        response = self.session.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            stories = []
            for item in data.get('reel', {}).get('items', []):
                story = self._parse_story(item)
                stories.append(story)
            
            return {
                'platform': 'instagram',
                'type': 'stories',
                'username': username,
                'stories': stories,
                'count': len(stories),
                'scraped_at': time.time(),
                'method': 'api'
            }
        
        raise Exception(f"Stories request failed: {response.status_code}")
    
    def _get_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username using web scraping"""
        headers = self.anti_detection.get_headers('instagram')
        response = self.session.get(f'{self.base_url}/{username}/', headers=headers)
        
        if response.status_code == 200:
            # Extract user ID from HTML
            match = re.search(r'"profilePage_([0-9]+)"', response.text)
            if match:
                return match.group(1)
            
            # Try another pattern
            match = re.search(r'"owner":\s*{\s*"id":\s*"([0-9]+)"', response.text)
            if match:
                return match.group(1)
        
        return None
    
    def _get_user_id_api(self, username: str) -> Optional[str]:
        """Get user ID using API"""
        url = f'{self.api_url}/users/{username}/usernameinfo/'
        
        headers = self.anti_detection.get_headers('instagram')
        headers.update({
            'User-Agent': 'Instagram 121.0.0.29.119 Android'
        })
        
        response = self.session.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('user', {}).get('pk')
        
        return None
    
    def _shortcode_to_media_id(self, shortcode: str) -> str:
        """Convert shortcode to media ID"""
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
        media_id = 0
        
        for char in shortcode:
            media_id = media_id * 64 + alphabet.index(char)
        
        return str(media_id)
    
    def _parse_graphql_post(self, data: Dict) -> Dict:
        """Parse post from GraphQL response"""
        post_data = {
            'id': data['id'],
            'shortcode': data['shortcode'],
            'caption': data.get('edge_media_to_caption', {}).get('edges', [{}])[0].get('node', {}).get('text', ''),
            'author': {
                'username': data['owner']['username'],
                'id': data['owner']['id']
            },
            'metrics': {
                'like_count': data.get('edge_media_preview_like', {}).get('count', 0),
                'comment_count': data.get('edge_media_to_comment', {}).get('count', 0)
            },
            'created_at': data.get('taken_at_timestamp'),
            'is_video': data.get('is_video', False),
            'video_view_count': data.get('video_view_count', 0)
        }
        
        # Media
        if data.get('edge_sidecar_to_children'):
            # Multiple media
            media = []
            for edge in data['edge_sidecar_to_children']['edges']:
                node = edge['node']
                media.append({
                    'type': 'video' if node.get('is_video') else 'image',
                    'url': node.get('video_url') if node.get('is_video') else node.get('display_url'),
                    'thumbnail': node.get('display_url')
                })
            post_data['media'] = media
        else:
            # Single media
            post_data['media'] = [{
                'type': 'video' if data.get('is_video') else 'image',
                'url': data.get('video_url') if data.get('is_video') else data.get('display_url'),
                'thumbnail': data.get('display_url')
            }]
        
        return post_data
    
    def _parse_api_post(self, data: Dict) -> Dict:
        """Parse post from API response"""
        post_data = {
            'id': data.get('id'),
            'shortcode': data.get('code'),
            'caption': data.get('caption', {}).get('text', ''),
            'author': {
                'username': data.get('user', {}).get('username'),
                'id': data.get('user', {}).get('pk')
            },
            'metrics': {
                'like_count': data.get('like_count', 0),
                'comment_count': data.get('comment_count', 0)
            },
            'created_at': data.get('taken_at'),
            'media_type': data.get('media_type')  # 1=photo, 2=video, 8=carousel
        }
        
        # Media
        media = []
        
        if data.get('carousel_media'):
            # Multiple media
            for item in data['carousel_media']:
                media.append({
                    'type': 'video' if item.get('media_type') == 2 else 'image',
                    'url': item.get('video_versions', [{}])[0].get('url') if item.get('media_type') == 2 else item.get('image_versions2', {}).get('candidates', [{}])[0].get('url')
                })
        else:
            # Single media
            if data.get('media_type') == 2:
                # Video
                media.append({
                    'type': 'video',
                    'url': data.get('video_versions', [{}])[0].get('url')
                })
            else:
                # Image
                media.append({
                    'type': 'image',
                    'url': data.get('image_versions2', {}).get('candidates', [{}])[0].get('url')
                })
        
        post_data['media'] = media
        
        return post_data
    
    def _parse_timeline_post(self, node: Dict) -> Dict:
        """Parse post from timeline"""
        return {
            'shortcode': node.get('shortcode'),
            'url': f"{self.base_url}/p/{node.get('shortcode')}/",
            'thumbnail': node.get('thumbnail_src') or node.get('display_url'),
            'is_video': node.get('is_video', False),
            'metrics': {
                'like_count': node.get('edge_liked_by', {}).get('count', 0),
                'comment_count': node.get('edge_media_to_comment', {}).get('count', 0)
            },
            'created_at': node.get('taken_at_timestamp')
        }
    
    def _parse_story(self, item: Dict) -> Dict:
        """Parse story from API response"""
        return {
            'id': item.get('id'),
            'media_type': 'video' if item.get('media_type') == 2 else 'image',
            'url': item.get('video_versions', [{}])[0].get('url') if item.get('media_type') == 2 else item.get('image_versions2', {}).get('candidates', [{}])[0].get('url'),
            'taken_at': item.get('taken_at'),
            'expiring_at': item.get('expiring_at'),
            'has_audio': item.get('has_audio', False)
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
        
        # Mobile emulation for better Instagram compatibility
        mobile_emulation = {
            "deviceMetrics": {"width": 360, "height": 640, "pixelRatio": 3.0},
            "userAgent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36 Instagram 195.0.0.31.123"
        }
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Create driver
        self.driver = webdriver.Chrome(options=options)
        
        # Inject stealth JavaScript
        stealth_js = self.anti_detection.get_stealth_js_injection()
        self.driver.execute_script(stealth_js)
    
    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        # Add random variation
        time.sleep(random.uniform(0.5, 1.5))
        self.last_request_time = time.time()
    
    def _parse_count(self, text: str) -> int:
        """Parse count from text (1.2k -> 1200)"""
        if not text:
            return 0
        
        text = text.strip().upper()
        
        # Remove commas
        text = text.replace(',', '')
        
        if 'K' in text:
            return int(float(text.replace('K', '')) * 1000)
        elif 'M' in text:
            return int(float(text.replace('M', '')) * 1000000)
        else:
            try:
                return int(text)
            except:
                return 0
    
    # Alternative service methods
    def _scrape_picuki_post(self, shortcode: str) -> Dict:
        """Scrape post from Picuki"""
        url = f'https://www.picuki.com/media/{shortcode}'
        headers = self.anti_detection.get_headers('generic')
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            post_data = {}
            
            # Media
            media = []
            media_box = soup.find('div', class_='single-photo')
            if media_box:
                img = media_box.find('img')
                if img:
                    media.append({
                        'type': 'image',
                        'url': img.get('src')
                    })
                
                video = media_box.find('video')
                if video:
                    media.append({
                        'type': 'video',
                        'url': video.get('src')
                    })
            
            post_data['media'] = media
            
            # Caption
            caption_elem = soup.find('div', class_='photo-description')
            if caption_elem:
                post_data['caption'] = caption_elem.get_text(strip=True)
            
            # Author
            author_elem = soup.find('div', class_='photo-info').find('a')
            if author_elem:
                post_data['author'] = {
                    'username': author_elem.text.replace('@', '')
                }
            
            # Metrics
            metrics = {}
            likes_elem = soup.find('div', class_='likes_photo')
            if likes_elem:
                metrics['like_count'] = self._parse_count(likes_elem.get_text())
            
            post_data['metrics'] = metrics
            
            return {
                'platform': 'instagram',
                'type': 'post',
                'shortcode': shortcode,
                'url': f'https://instagram.com/p/{shortcode}/',
                'data': post_data,
                'scraped_at': time.time(),
                'method': 'picuki'
            }
        
        raise Exception(f"Picuki request failed: {response.status_code}")
    
    def _scrape_imginn_post(self, shortcode: str) -> Dict:
        """Scrape post from Imginn"""
        url = f'https://imginn.com/p/{shortcode}/'
        headers = self.anti_detection.get_headers('generic')
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            post_data = {}
            
            # Extract JSON-LD data
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    data = json.loads(json_ld.string)
                    
                    post_data['author'] = {
                        'username': data.get('author', {}).get('alternateName', '').replace('@', '')
                    }
                    
                    post_data['caption'] = data.get('articleBody', '')
                    
                    # Media
                    media = []
                    if 'image' in data:
                        images = data['image'] if isinstance(data['image'], list) else [data['image']]
                        for img in images:
                            media.append({
                                'type': 'image',
                                'url': img
                            })
                    
                    if 'video' in data:
                        videos = data['video'] if isinstance(data['video'], list) else [data['video']]
                        for vid in videos:
                            media.append({
                                'type': 'video',
                                'url': vid.get('contentUrl', vid) if isinstance(vid, dict) else vid
                            })
                    
                    post_data['media'] = media
                    
                except:
                    pass
            
            return {
                'platform': 'instagram',
                'type': 'post',
                'shortcode': shortcode,
                'url': f'https://instagram.com/p/{shortcode}/',
                'data': post_data,
                'scraped_at': time.time(),
                'method': 'imginn'
            }
        
        raise Exception(f"Imginn request failed: {response.status_code}")
    
    def _scrape_pixwox_post(self, shortcode: str) -> Dict:
        """Scrape post from Pixwox"""
        url = f'https://www.pixwox.com/post/{shortcode}/'
        headers = self.anti_detection.get_headers('generic')
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            post_data = {}
            
            # Similar parsing logic as other services
            # ... (implementation details)
            
            return {
                'platform': 'instagram',
                'type': 'post',
                'shortcode': shortcode,
                'url': f'https://instagram.com/p/{shortcode}/',
                'data': post_data,
                'scraped_at': time.time(),
                'method': 'pixwox'
            }
        
        raise Exception(f"Pixwox request failed: {response.status_code}")
    
    def _scrape_picuki_user(self, username: str, limit: int) -> Dict:
        """Scrape user posts from Picuki"""
        url = f'https://www.picuki.com/profile/{username}'
        headers = self.anti_detection.get_headers('generic')
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            posts = []
            
            # Find all post boxes
            post_boxes = soup.find_all('div', class_='box-photo')[:limit]
            
            for box in post_boxes:
                link = box.find('a')
                if link:
                    href = link.get('href')
                    shortcode = href.split('/media/')[-1] if '/media/' in href else None
                    
                    if shortcode:
                        post_data = {
                            'shortcode': shortcode,
                            'url': f'https://instagram.com/p/{shortcode}/'
                        }
                        
                        # Thumbnail
                        img = box.find('img')
                        if img:
                            post_data['thumbnail'] = img.get('src')
                        
                        posts.append(post_data)
            
            return {
                'platform': 'instagram',
                'type': 'user_posts',
                'username': username,
                'posts': posts,
                'count': len(posts),
                'scraped_at': time.time(),
                'method': 'picuki'
            }
        
        raise Exception(f"Picuki user request failed: {response.status_code}")
    
    def _scrape_imginn_user(self, username: str, limit: int) -> Dict:
        """Scrape user posts from Imginn"""
        # Similar implementation as Picuki
        pass
    
    def _scrape_pixwox_user(self, username: str, limit: int) -> Dict:
        """Scrape user posts from Pixwox"""
        # Similar implementation as Picuki
        pass
    
    def _scrape_hashtag_web(self, hashtag: str, limit: int) -> Dict:
        """Scrape hashtag using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f'{self.base_url}/explore/tags/{hashtag}/'
        self.driver.get(url)
        
        # Wait for posts to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'article a[href*="/p/"]'))
            )
        except TimeoutException:
            raise Exception("No posts found or timeout")
        
        posts = []
        seen_shortcodes = set()
        
        # Get top posts first
        try:
            top_posts_section = self.driver.find_element(By.XPATH, '//h2[contains(text(), "Top posts")]/parent::div')
            top_links = top_posts_section.find_elements(By.CSS_SELECTOR, 'a[href*="/p/"]')
            
            for link in top_links[:9]:  # Usually 9 top posts
                href = link.get_attribute('href')
                shortcode = self._extract_shortcode(href)
                
                if shortcode and shortcode not in seen_shortcodes:
                    seen_shortcodes.add(shortcode)
                    posts.append({
                        'shortcode': shortcode,
                        'url': href,
                        'is_top_post': True
                    })
        except:
            pass
        
        # Get recent posts
        last_height = 0
        no_change_count = 0
        
        while len(posts) < limit and no_change_count < 3:
            # Get all post links
            all_links = self.driver.find_elements(By.CSS_SELECTOR, 'article a[href*="/p/"]')
            
            for link in all_links:
                if len(posts) >= limit:
                    break
                
                href = link.get_attribute('href')
                shortcode = self._extract_shortcode(href)
                
                if shortcode and shortcode not in seen_shortcodes:
                    seen_shortcodes.add(shortcode)
                    posts.append({
                        'shortcode': shortcode,
                        'url': href,
                        'is_top_post': False
                    })
            
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
            'platform': 'instagram',
            'type': 'hashtag',
            'hashtag': hashtag,
            'posts': posts[:limit],
            'count': len(posts),
            'scraped_at': time.time(),
            'method': 'web'
        }
    
    def _scrape_hashtag_alt(self, hashtag: str, limit: int) -> Dict:
        """Scrape hashtag using alternative services"""
        # Similar to user posts alternative scraping
        pass
    
    def _scrape_stories_web(self, username: str) -> Dict:
        """Scrape stories using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        # Navigate to user profile
        url = f'{self.base_url}/{username}/'
        self.driver.get(url)
        
        # Check if user has stories
        try:
            story_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="button"] canvas'))
            )
            
            # Click on story
            story_button.click()
            
            time.sleep(2)
            
            stories = []
            story_count = 0
            
            while True:
                try:
                    # Get current story
                    story_elem = self.driver.find_element(By.CSS_SELECTOR, 'div[role="presentation"] img, div[role="presentation"] video')
                    
                    story_data = {
                        'index': story_count,
                        'type': 'video' if story_elem.tag_name == 'video' else 'image',
                        'url': story_elem.get_attribute('src')
                    }
                    
                    stories.append(story_data)
                    story_count += 1
                    
                    # Click next
                    next_button = self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]')
                    next_button.click()
                    time.sleep(1)
                    
                except:
                    # No more stories
                    break
            
            return {
                'platform': 'instagram',
                'type': 'stories',
                'username': username,
                'stories': stories,
                'count': len(stories),
                'scraped_at': time.time(),
                'method': 'web'
            }
            
        except TimeoutException:
            # No stories available
            return {
                'platform': 'instagram',
                'type': 'stories',
                'username': username,
                'stories': [],
                'count': 0,
                'scraped_at': time.time(),
                'method': 'web'
            }
    
    def __del__(self):
        """Cleanup"""
        if self.driver:
            self.driver.quit()# scrapers/instagram_scraper.py
"""
Instagram Scraper with anti-detection
Supports GraphQL API, web scraping, and mobile API endpoints
"""

import re
import json
import time
import random
import hashlib
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote, urlencode
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from utils.anti_detection import AntiDetection

class InstagramScraper:
    """
    Instagram scraper with multiple methods:
    - GraphQL API (requires session)
    - Web scraping with Selenium
    - Mobile API endpoints
    - Picuki/Imginn fallback
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'instagram_scraper.{node_id}')
        
        # Anti-detection
        self.anti_detection = AntiDetection()
        
        # Session management
        self.session = requests.Session()
        self.driver = None
        
        # Instagram API endpoints
        self.base_url = 'https://www.instagram.com'
        self.api_url = 'https://i.instagram.com/api/v1'
        self.graphql_url = f'{self.base_url}/graphql/query'
        
        # Authentication
        self.session_id = None
        self.csrf_token = None
        self.user_agent = None
        
        # Query hashes for GraphQL
        self.query_hashes = {
            'user_posts': 'e769aa130647d2354c40ea37b90b7c8',
            'post_details': '2b0673e0dc4580674a88d426fe00ea90',
            'user_info': 'd4d88dc1500312af6f937f7b804c68c3',
            'hashtag_posts': 'f92f56d47dc7a55b606908374b43a314',
            'location_posts': '1b84447a4d8b6d6d0426fefb34514485',
            'stories': 'de8017ee0a7c9c45ec4260733d81ea31',
            'highlights': 'd4d88dc1500312af6f937f7b804c68c3',
            'comments': 'bc3296d1ce80a24b1b6e40b1e72903f5',
            'likers': 'd5d763b1e2acf209d62