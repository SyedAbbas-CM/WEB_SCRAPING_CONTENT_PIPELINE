# utils/proxy_providers.py
"""
Integration with multiple proxy providers
Supports residential, datacenter, and mobile proxies
"""

import os
import json
import time
import random
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class ProxyEndpoint:
    """Proxy endpoint configuration"""
    provider: str
    type: str  # residential, datacenter, mobile
    endpoint: str
    port: int
    username: str
    password: str
    country: Optional[str] = None
    city: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_url(self) -> str:
        """Convert to proxy URL"""
        if self.username and self.password:
            return f"http://{self.username}:{self.password}@{self.endpoint}:{self.port}"
        return f"http://{self.endpoint}:{self.port}"

class ProxyProvider:
    """
    Manages multiple proxy providers and automatic rotation
    """
    
    def __init__(self):
        self.logger = logging.getLogger('proxy_provider')
        self.providers = self._init_providers()
        self.active_proxies = []
        self.failed_proxies = []
        
    def _init_providers(self) -> Dict:
        """Initialize proxy provider configurations"""
        providers = {}
        
        # BrightData (formerly Luminati)
        if os.getenv('BRIGHTDATA_USERNAME'):
            providers['brightdata'] = {
                'residential': {
                    'endpoint': 'pr.oxylabs.io',
                    'port': 7777,
                    'username': os.getenv('OXYLABS_USERNAME'),
                    'password': os.getenv('OXYLABS_PASSWORD'),
                    'session_support': True,
                    'sticky_sessions': True
                },
                'datacenter': {
                    'endpoint': 'dc.oxylabs.io',
                    'port': 8001,
                    'username': os.getenv('OXYLABS_USERNAME'),
                    'password': os.getenv('OXYLABS_PASSWORD'),
                    'session_support': False
                }
            }
        
        # Free proxy lists (fallback only)
        providers['free'] = {
            'mixed': {
                'api_endpoints': [
                    'https://www.proxy-list.download/api/v1/get?type=http',
                    'https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all'
                ],
                'update_interval': 3600  # Update hourly
            }
        }
        
        return providers
    
    def get_proxy(self, proxy_type: str = 'residential', 
                  country: Optional[str] = None,
                  session: bool = False) -> Optional[ProxyEndpoint]:
        """Get a proxy of specified type"""
        
        # Try each provider in order of preference
        provider_order = ['brightdata', 'oxylabs', 'smartproxy', 'proxymesh', 'free']
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
                
            provider = self.providers[provider_name]
            if proxy_type not in provider:
                continue
                
            try:
                proxy = self._create_proxy_from_provider(
                    provider_name,
                    provider[proxy_type],
                    proxy_type,
                    country,
                    session
                )
                
                if proxy and self._test_proxy(proxy):
                    self.active_proxies.append(proxy)
                    return proxy
                    
            except Exception as e:
                self.logger.error(f"Failed to get proxy from {provider_name}: {e}")
                continue
        
        self.logger.error(f"No available {proxy_type} proxies")
        return None
    
    def _create_proxy_from_provider(self, provider_name: str, config: Dict,
                                  proxy_type: str, country: Optional[str],
                                  session: bool) -> ProxyEndpoint:
        """Create proxy endpoint from provider config"""
        
        # Handle different provider formats
        if provider_name == 'brightdata':
            username = config['username']
            
            # Add session ID if requested
            if session and config.get('session_support'):
                session_id = f"session-{int(time.time() * 1000)}"
                username = f"{username}-session-{session_id}"
            
            # Add country if specified
            if country and country in config.get('countries', []):
                username = f"{username}-country-{country}"
            
            return ProxyEndpoint(
                provider=provider_name,
                type=proxy_type,
                endpoint=config['endpoint'],
                port=config['port'],
                username=username,
                password=config['password'],
                country=country,
                session_id=session_id if session else None
            )
            
        elif provider_name == 'smartproxy':
            # SmartProxy uses different ports for sticky sessions
            port = config['port'] if session else 10001  # Random port
            
            return ProxyEndpoint(
                provider=provider_name,
                type=proxy_type,
                endpoint=config['endpoint'],
                port=port,
                username=config['username'],
                password=config['password']
            )
            
        elif provider_name == 'proxymesh':
            # ProxyMesh has multiple endpoints
            endpoint_full = random.choice(config['endpoints'])
            endpoint, port = endpoint_full.split(':')
            
            return ProxyEndpoint(
                provider=provider_name,
                type=proxy_type,
                endpoint=endpoint,
                port=int(port),
                username=config['username'],
                password=config['password']
            )
            
        elif provider_name == 'free':
            # Fetch free proxy from API
            proxy = self._get_free_proxy()
            if proxy:
                return ProxyEndpoint(
                    provider=provider_name,
                    type='unknown',
                    endpoint=proxy['ip'],
                    port=proxy['port'],
                    username='',
                    password=''
                )
        
        raise ValueError(f"Unknown provider: {provider_name}")
    
    def _get_free_proxy(self) -> Optional[Dict]:
        """Get a free proxy from public lists"""
        for api_endpoint in self.providers['free']['mixed']['api_endpoints']:
            try:
                response = requests.get(api_endpoint, timeout=10)
                
                if 'proxy-list.download' in api_endpoint:
                    # Parse proxy list format
                    proxies = response.text.strip().split('\n')
                    if proxies:
                        proxy = random.choice(proxies)
                        ip, port = proxy.split(':')
                        return {'ip': ip, 'port': int(port)}
                        
                elif 'proxyscrape' in api_endpoint:
                    # Parse ProxyScrape format
                    proxies = response.text.strip().split('\r\n')
                    if proxies:
                        proxy = random.choice(proxies)
                        ip, port = proxy.split(':')
                        return {'ip': ip, 'port': int(port)}
                        
            except Exception as e:
                self.logger.debug(f"Failed to fetch free proxies: {e}")
                continue
        
        return None
    
    def _test_proxy(self, proxy: ProxyEndpoint, timeout: int = 10) -> bool:
        """Test if proxy is working"""
        test_url = 'http://httpbin.org/ip'
        
        try:
            response = requests.get(
                test_url,
                proxies={
                    'http': proxy.to_url(),
                    'https': proxy.to_url()
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                # Verify we're using the proxy
                detected_ip = response.json().get('origin', '')
                self.logger.debug(f"Proxy test successful: {detected_ip}")
                return True
                
        except Exception as e:
            self.logger.debug(f"Proxy test failed for {proxy.endpoint}: {e}")
        
        return False
    
    def mark_proxy_failed(self, proxy: ProxyEndpoint, reason: str = "unknown"):
        """Mark a proxy as failed"""
        self.failed_proxies.append({
            'proxy': proxy,
            'reason': reason,
            'failed_at': time.time()
        })
        
        # Remove from active proxies
        self.active_proxies = [p for p in self.active_proxies if p != proxy]
        
        self.logger.warning(
            f"Marked proxy as failed: {proxy.provider} {proxy.type} - {reason}"
        )
    
    def get_proxy_stats(self) -> Dict:
        """Get statistics about proxy pools"""
        stats = {
            'providers': list(self.providers.keys()),
            'active_proxies': len(self.active_proxies),
            'failed_proxies': len(self.failed_proxies),
            'by_type': {},
            'by_provider': {}
        }
        
        # Count by type
        for proxy in self.active_proxies:
            stats['by_type'][proxy.type] = stats['by_type'].get(proxy.type, 0) + 1
            stats['by_provider'][proxy.provider] = stats['by_provider'].get(proxy.provider, 0) + 1
        
        return stats
    
    def rotate_all_sessions(self):
        """Rotate all session-based proxies"""
        new_active = []
        
        for proxy in self.active_proxies:
            if proxy.session_id:
                # Create new session
                new_proxy = self.get_proxy(
                    proxy_type=proxy.type,
                    country=proxy.country,
                    session=True
                )
                if new_proxy:
                    new_active.append(new_proxy)
            else:
                # Keep non-session proxies
                new_active.append(proxy)
        
        self.active_proxies = new_active
        self.logger.info(f"Rotated sessions, {len(new_active)} proxies active")


# Example configuration file: config/proxy_config.json
EXAMPLE_PROXY_CONFIG = {
    "providers": {
        "brightdata": {
            "username": "your_brightdata_username",
            "password": "your_brightdata_password",
            "enabled": True,
            "preferred_for": ["instagram", "facebook"]
        },
        "smartproxy": {
            "username": "your_smartproxy_username",
            "password": "your_smartproxy_password",
            "enabled": True,
            "preferred_for": ["reddit", "twitter"]
        },
        "oxylabs": {
            "username": "your_oxylabs_username",
            "password": "your_oxylabs_password",
            "enabled": False
        }
    },
    "rotation_rules": {
        "max_requests_per_proxy": 100,
        "max_age_minutes": 30,
        "rotate_on_ban": True,
        "rotate_on_captcha": True
    },
    "platform_preferences": {
        "reddit": {
            "proxy_types": ["datacenter", "residential"],
            "require_session": False,
            "countries": ["us", "ca", "uk"]
        },
        "instagram": {
            "proxy_types": ["mobile", "residential"],
            "require_session": True,
            "countries": ["us"]
        },
        "tiktok": {
            "proxy_types": ["mobile"],
            "require_session": True,
            "countries": ["us"]
        }
    }
} {
                    'endpoint': 'zproxy.lum-superproxy.io',
                    'port': 22225,
                    'username': os.getenv('BRIGHTDATA_USERNAME'),
                    'password': os.getenv('BRIGHTDATA_PASSWORD'),
                    'session_support': True,
                    'countries': ['us', 'uk', 'ca', 'au', 'de', 'fr']
                },
                'datacenter': {
                    'endpoint': 'zproxy.lum-superproxy.io',
                    'port': 22225,
                    'username': os.getenv('BRIGHTDATA_USERNAME') + '-dc',
                    'password': os.getenv('BRIGHTDATA_PASSWORD'),
                    'session_support': False
                },
                'mobile': {
                    'endpoint': 'mobile.lum-superproxy.io',
                    'port': 22225,
                    'username': os.getenv('BRIGHTDATA_USERNAME') + '-mobile',
                    'password': os.getenv('BRIGHTDATA_PASSWORD'),
                    'session_support': True,
                    'carriers': ['verizon', 'att', 'tmobile']
                }
            }
        
        # SmartProxy
        if os.getenv('SMARTPROXY_USERNAME'):
            providers['smartproxy'] = {
                'residential': {
                    'endpoint': 'gate.smartproxy.com',
                    'port': 10000,  # Sticky port
                    'username': os.getenv('SMARTPROXY_USERNAME'),
                    'password': os.getenv('SMARTPROXY_PASSWORD'),
                    'session_support': True
                },
                'datacenter': {
                    'endpoint': 'gate.dc.smartproxy.com',
                    'port': 10000,
                    'username': os.getenv('SMARTPROXY_USERNAME'),
                    'password': os.getenv('SMARTPROXY_PASSWORD'),
                    'session_support': False
                }
            }
        
        # ProxyMesh
        if os.getenv('PROXYMESH_USERNAME'):
            providers['proxymesh'] = {
                'datacenter': {
                    'endpoints': [
                        'us-wa.proxymesh.com:31280',
                        'us-ca.proxymesh.com:31280',
                        'us-il.proxymesh.com:31280',
                        'us-ny.proxymesh.com:31280'
                    ],
                    'username': os.getenv('PROXYMESH_USERNAME'),
                    'password': os.getenv('PROXYMESH_PASSWORD'),
                    'rotate_on_request': True
                }
            }
        
        # Oxylabs
        if os.getenv('OXYLABS_USERNAME'):
            providers['oxylabs'] = {
                'residential':