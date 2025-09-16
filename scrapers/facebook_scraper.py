"""
Facebook scraper: compliant Graph API when token provided, minimal web fallback for public pages.
"""

import os
import time
import json
import requests
from typing import Dict
from urllib.parse import urlencode
import logging

from utils.anti_detection import AntiDetection

class FacebookScraper:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'facebook_scraper.{node_id}')
        self.session = requests.Session()
        self.anti = AntiDetection()
        self.access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
        self.graph_base = 'https://graph.facebook.com/v18.0'
    
    def set_proxy(self, proxy: Dict):
        if proxy and proxy.get('url'):
            self.session.proxies = {'http': proxy['url'], 'https': proxy['url']}
    
    def set_headers(self, headers: Dict):
        self.session.headers.update(headers)
    
    def get_page_posts(self, page_id_or_name: str, limit: int = 25) -> Dict:
        mode = os.getenv('COMPLIANCE_MODE', 'api_first').lower()
        if self.access_token and mode in ['api_only', 'api_first']:
            return self._get_page_posts_api(page_id_or_name, limit)
        return self._get_page_posts_web(page_id_or_name, limit)
    
    def _get_page_posts_api(self, page: str, limit: int) -> Dict:
        params = {'access_token': self.access_token, 'limit': limit, 'fields': 'id,message,created_time,permalink_url,full_picture'}
        url = f'{self.graph_base}/{page}/posts?{urlencode(params)}'
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = []
        for p in data.get('data', []):
            items.append({
                'id': p.get('id'),
                'message': p.get('message',''),
                'created_time': p.get('created_time'),
                'permalink_url': p.get('permalink_url'),
                'full_picture': p.get('full_picture')
            })
        return {'platform': 'facebook', 'type': 'page_posts', 'page': page, 'results': items, 'count': len(items), 'method': 'api'}
    
    def _get_page_posts_web(self, page: str, limit: int) -> Dict:
        headers = self.anti.get_headers('generic')
        url = f'https://m.facebook.com/{page}/posts'
        r = self.session.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        # Minimal placeholder: parsing FB HTML reliably is complex; extract basic links
        results = []
        for line in r.text.split('\n'):
            if '/story.php?' in line and 'href="' in line and len(results) < limit:
                try:
                    href = line.split('href="',1)[1].split('"',1)[0]
                    results.append({'url': 'https://m.facebook.com' + href})
                except Exception:
                    pass
        return {'platform': 'facebook', 'type': 'page_posts', 'page': page, 'results': results, 'count': len(results), 'method': 'web'}

