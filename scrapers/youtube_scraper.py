"""
Minimal YouTube scraper supporting compliant API mode and basic web fallback.
"""

import os
import time
import json
import requests
from typing import Dict, Optional
from urllib.parse import urlencode
import logging

from utils.anti_detection import AntiDetection

class YouTubeScraper:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'youtube_scraper.{node_id}')
        self.session = requests.Session()
        self.anti = AntiDetection()
        self.api_key = os.getenv('YOUTUBE_API_KEY')
    
    def set_proxy(self, proxy: Dict):
        if proxy and proxy.get('url'):
            self.session.proxies = {'http': proxy['url'], 'https': proxy['url']}
    
    def set_headers(self, headers: Dict):
        self.session.headers.update(headers)
    
    def search(self, query: str, max_results: int = 25) -> Dict:
        mode = os.getenv('COMPLIANCE_MODE', 'api_first').lower()
        if self.api_key and mode in ['api_only', 'api_first']:
            return self._search_api(query, max_results)
        return self._search_web(query, max_results)
    
    def _search_api(self, query: str, max_results: int) -> Dict:
        params = {
            'part': 'snippet',
            'q': query,
            'maxResults': max_results,
            'type': 'video',
            'key': self.api_key
        }
        url = f'https://www.googleapis.com/youtube/v3/search?{urlencode(params)}'
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = []
        for item in data.get('items', []):
            vid = item['id']['videoId']
            snip = item['snippet']
            items.append({
                'id': vid,
                'title': snip['title'],
                'channel': snip['channelTitle'],
                'publishedAt': snip['publishedAt'],
                'thumbnail': snip['thumbnails']['high']['url']
            })
        return {'platform': 'youtube', 'type': 'search', 'query': query, 'results': items, 'count': len(items), 'method': 'api'}
    
    def _search_web(self, query: str, max_results: int) -> Dict:
        headers = self.anti.get_headers('generic')
        url = f'https://www.youtube.com/results?search_query={query}'
        r = self.session.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        text = r.text
        # Extract ytInitialData
        marker = 'var ytInitialData = '
        idx = text.find(marker)
        results = []
        if idx != -1:
            end = text.find(';</script>', idx)
            if end != -1:
                blob = text[idx + len(marker):end]
                try:
                    data = json.loads(blob)
                    contents = data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents']
                    for section in contents:
                        items = section.get('itemSectionRenderer', {}).get('contents', [])
                        for it in items:
                            vid = it.get('videoRenderer')
                            if vid and len(results) < max_results:
                                results.append({
                                    'id': vid.get('videoId'),
                                    'title': ''.join([r.get('text','') for r in vid.get('title',{}).get('runs',[])]),
                                    'channel': ''.join([r.get('text','') for r in vid.get('ownerText',{}).get('runs',[])]),
                                    'thumbnail': vid.get('thumbnail',{}).get('thumbnails',[{}])[-1].get('url')
                                })
                except Exception:
                    pass
        return {'platform': 'youtube', 'type': 'search', 'query': query, 'results': results, 'count': len(results), 'method': 'web'}

