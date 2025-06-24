# nodes/scraping_node.py
"""
Dedicated Scraping Node
Handles all web scraping with anti-detection measures
"""

import os
import time
import json
import random
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .base_node import BaseNode, NodeType, Task
from scrapers.reddit_scraper import RedditScraper
from scrapers.twitter_scraper import TwitterScraper
from scrapers.instagram_scraper import InstagramScraper
from scrapers.tiktok_scraper import TikTokScraper
from utils.anti_detection import AntiDetection
from utils.proxy_manager import ProxyManager
from utils.rate_limiter import RateLimiter

class ScrapingNode(BaseNode):
    """
    Dedicated scraping node with:
    - Multi-platform support
    - Anti-detection measures
    - Proxy rotation
    - Rate limiting
    - Session management
    """
    
    def __init__(self, node_id: str, master_ip: str = None, capabilities: List[str] = None):
        super().__init__(node_id, NodeType.SCRAPING_NODE, master_ip)
        
        # Scraping capabilities
        self.capabilities = capabilities or ['reddit', 'twitter']
        self.logger.info(f"Scraping capabilities: {self.capabilities}")
        
        # Initialize components
        self.scrapers = self._init_scrapers()
        self.proxy_manager = ProxyManager()
        self.rate_limiter = RateLimiter()
        self.anti_detection = AntiDetection()
        
        # Session storage
        self.sessions = {}
        
        # Performance metrics
        self.scraping_metrics = {
            'total_scraped': 0,
            'success_rate': 0,
            'blocked_count': 0,
            'proxy_rotations': 0
        }
        
    def _init_scrapers(self) -> Dict:
        """Initialize platform-specific scrapers"""
        scrapers = {}
        
        if 'reddit' in self.capabilities:
            scrapers['reddit'] = RedditScraper(self.node_id)
            self.logger.info("Initialized Reddit scraper")
            
        if 'twitter' in self.capabilities:
            scrapers['twitter'] = TwitterScraper(self.node_id)
            self.logger.info("Initialized Twitter scraper")
            
        if 'instagram' in self.capabilities:
            scrapers['instagram'] = InstagramScraper(self.node_id)
            self.logger.info("Initialized Instagram scraper")
            
        if 'tiktok' in self.capabilities:
            scrapers['tiktok'] = TikTokScraper(self.node_id)
            self.logger.info("Initialized TikTok scraper")
            
        if 'browser' in self.capabilities:
            self._setup_browser_automation()
            
        return scrapers
    
    def _setup_browser_automation(self):
        """Setup Selenium/Playwright for heavy scraping"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import undetected_chromedriver as uc
            
            self.browser_available = True
            self.logger.info("Browser automation available")
        except ImportError:
            self.browser_available = False
            self.logger.warning("Browser automation not available")
    
    def get_node_info(self) -> Dict:
        """Add scraping-specific info"""
        info = super().get_node_info()
        info['capabilities'] = self.capabilities
        info['scraping_metrics'] = self.scraping_metrics
        info['active_proxies'] = self.proxy_manager.get_active_count()
        info['rate_limits'] = self.rate_limiter.get_current_limits()
        return info
    
    def process_task(self, task: Task) -> Dict:
        """Process scraping task"""
        url = task.target
        platform = self._detect_platform(url)
        
        # Check rate limits
        if not self.rate_limiter.can_scrape(platform):
            wait_time = self.rate_limiter.get_wait_time(platform)
            self.logger.warning(f"Rate limit hit for {platform}, waiting {wait_time}s")
            time.sleep(wait_time)
        
        # Get proxy
        proxy = self.proxy_manager.get_proxy(platform)
        
        # Apply anti-detection measures
        headers = self.anti_detection.get_headers(platform)
        
        # Scrape
        try:
            result = self._scrape(url, platform, proxy, headers, task.metadata)
            self.scraping_metrics['success_rate'] = (
                self.scraping_metrics['success_rate'] * self.scraping_metrics['total_scraped'] + 1
            ) / (self.scraping_metrics['total_scraped'] + 1)
            self.scraping_metrics['total_scraped'] += 1
            
            # Queue for analysis if needed
            if task.metadata and task.metadata.get('analyze', True):
                self._queue_for_analysis(result)
            
            return result
            
        except BlockedException as e:
            self.logger.error(f"Blocked on {platform}: {e}")
            self.scraping_metrics['blocked_count'] += 1
            
            # Rotate proxy
            self.proxy_manager.mark_bad_proxy(proxy)
            self.scraping_metrics['proxy_rotations'] += 1
            
            # Retry with new proxy
            new_proxy = self.proxy_manager.get_proxy(platform, exclude=[proxy])
            return self._scrape(url, platform, new_proxy, headers, task.metadata)
            
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            raise
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'reddit.com' in domain:
            return 'reddit'
        elif 'twitter.com' in domain or 'x.com' in domain:
            return 'twitter'
        elif 'instagram.com' in domain:
            return 'instagram'
        elif 'tiktok.com' in domain:
            return 'tiktok'
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        else:
            return 'generic'
    
    def _scrape(self, url: str, platform: str, proxy: Dict, headers: Dict, metadata: Dict) -> Dict:
        """Execute scraping with platform-specific scraper"""
        if platform not in self.scrapers:
            return self._generic_scrape(url, proxy, headers)
        
        scraper = self.scrapers[platform]
        
        # Configure scraper
        scraper.set_proxy(proxy)
        scraper.set_headers(headers)
        
        # Add delay for human-like behavior
        delay = random.uniform(
            float(os.getenv('MIN_SCRAPE_DELAY', 2)),
            float(os.getenv('MAX_SCRAPE_DELAY', 5))
        )
        time.sleep(delay)
        
        # Scrape based on URL pattern
        if platform == 'reddit':
            if '/comments/' in url:
                return scraper.scrape_post(url, include_comments=True)
            elif '/user/' in url:
                return scraper.scrape_user(url)
            else:
                return scraper.scrape_subreddit(url, limit=metadata.get('limit', 25))
                
        elif platform == 'twitter':
            if '/status/' in url:
                return scraper.scrape_tweet(url)
            elif '/@' in url or '/user/' in url:
                return scraper.scrape_user_timeline(url, limit=metadata.get('limit', 50))
            else:
                return scraper.scrape_search(url, metadata)
                
        elif platform == 'instagram':
            if '/p/' in url:
                return scraper.scrape_post(url)
            elif '/reel/' in url:
                return scraper.scrape_reel(url)
            else:
                return scraper.scrape_profile(url, limit=metadata.get('limit', 20))
                
        elif platform == 'tiktok':
            if '/video/' in url:
                return scraper.scrape_video(url)
            elif '/@' in url:
                return scraper.scrape_user(url, limit=metadata.get('limit', 30))
            else:
                return scraper.scrape_trending(metadata)
                
        else:
            return self._generic_scrape(url, proxy, headers)
    
    def _generic_scrape(self, url: str, proxy: Dict, headers: Dict) -> Dict:
        """Generic scraping for unknown platforms"""
        import requests
        from bs4 import BeautifulSoup
        
        # Configure session
        session = requests.Session()
        session.headers.update(headers)
        
        if proxy:
            session.proxies = {
                'http': proxy['url'],
                'https': proxy['url']
            }
        
        # Fetch page
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic info
        result = {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'meta_description': '',
            'text_content': soup.get_text()[:5000],  # First 5000 chars
            'images': [img.get('src') for img in soup.find_all('img')[:10]],
            'links': [a.get('href') for a in soup.find_all('a')[:20]],
            'scraped_at': time.time(),
            'status_code': response.status_code
        }
        
        # Try to get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            result['meta_description'] = meta_desc.get('content', '')
        
        return result
    
    def _queue_for_analysis(self, scraped_data: Dict):
        """Queue scraped data for AI analysis"""
        ai_task = {
            'id': f"analyze_{scraped_data.get('id', int(time.time()))}",
            'type': 'analyze_content',
            'target': 'content_analysis',
            'priority': 1,
            'status': 'pending',
            'created_at': time.time(),
            'metadata': {
                'content': scraped_data,
                'platform': scraped_data.get('platform', 'unknown'),
                'source_url': scraped_data.get('url', '')
            }
        }
        
        # Queue to AI node
        self.redis.lpush('ai_tasks:ai-node-01', json.dumps(ai_task))
        self.logger.info(f"Queued content for AI analysis: {ai_task['id']}")

class BlockedException(Exception):
    """Raised when scraping is blocked"""
    pass