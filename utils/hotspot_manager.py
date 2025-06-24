# utils/hotspot_manager.py
"""
Mobile Hotspot Manager for IP Rotation
Manages multiple 4G/5G hotspots for fresh IPs
"""

import os
import time
import subprocess
import platform
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class Hotspot:
    """Represents a mobile hotspot device"""
    id: str
    name: str
    interface: str  # wlan0, wlan1, etc.
    carrier: str    # verizon, att, tmobile
    phone_number: Optional[str] = None
    imei: Optional[str] = None
    current_ip: Optional[str] = None
    last_rotation: float = 0
    usage_count: int = 0
    blocked: bool = False

class HotspotManager:
    """
    Manages multiple mobile hotspots for IP rotation
    Supports USB tethering and WiFi hotspots
    """
    
    def __init__(self):
        self.logger = logging.getLogger('hotspot_manager')
        self.hotspots = self._detect_hotspots()
        self.active_hotspot = None
        self.rotation_interval = 1800  # 30 minutes
        
    def _detect_hotspots(self) -> Dict[str, Hotspot]:
        """Detect available hotspot devices"""
        hotspots = {}
        
        if platform.system() == 'Linux':
            # Detect USB tethered devices
            try:
                # Check network interfaces
                result = subprocess.run(['ip', 'link', 'show'], 
                                      capture_output=True, text=True)
                
                interfaces = result.stdout
                
                # Look for USB tethering interfaces (usually usb0, usb1, etc.)
                for i in range(5):
                    if f'usb{i}' in interfaces:
                        hotspot = Hotspot(
                            id=f'hotspot_usb_{i}',
                            name=f'USB Hotspot {i}',
                            interface=f'usb{i}',
                            carrier='unknown'
                        )
                        hotspots[hotspot.id] = hotspot
                        self.logger.info(f"Detected USB hotspot: {hotspot.interface}")
                
                # Look for WiFi hotspots
                # Check if connected to mobile hotspot SSIDs
                wifi_check = subprocess.run(['nmcli', 'device', 'wifi', 'list'],
                                          capture_output=True, text=True)
                
                # Common mobile hotspot patterns
                hotspot_patterns = ['iPhone', 'Android', 'Mobile Hotspot', 'Galaxy', 'Pixel']
                
                for line in wifi_check.stdout.split('\n'):
                    for pattern in hotspot_patterns:
                        if pattern in line and '*' in line:  # * indicates connected
                            hotspot = Hotspot(
                                id='hotspot_wifi_0',
                                name='WiFi Hotspot',
                                interface='wlan0',
                                carrier='unknown'
                            )
                            hotspots[hotspot.id] = hotspot
                            
            except Exception as e:
                self.logger.error(f"Failed to detect hotspots: {e}")
        
        elif platform.system() == 'Windows':
            # Windows mobile hotspot detection
            try:
                # Check for mobile broadband adapters
                result = subprocess.run(['netsh', 'mbn', 'show', 'interfaces'],
                                      capture_output=True, text=True, shell=True)
                
                if 'Mobile Broadband' in result.stdout:
                    hotspot = Hotspot(
                        id='hotspot_mbn_0',
                        name='Mobile Broadband',
                        interface='Mobile Broadband',
                        carrier='unknown'
                    )
                    hotspots[hotspot.id] = hotspot
                    
            except Exception as e:
                self.logger.error(f"Windows hotspot detection failed: {e}")
        
        return hotspots
    
    def get_current_ip(self, interface: str) -> Optional[str]:
        """Get current IP address of interface"""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(
                    ['ip', 'addr', 'show', interface],
                    capture_output=True, text=True
                )
                # Parse IP from output
                for line in result.stdout.split('\n'):
                    if 'inet ' in line:
                        ip = line.split()[1].split('/')[0]
                        return ip
                        
            elif platform.system() == 'Windows':
                result = subprocess.run(
                    ['ipconfig'],
                    capture_output=True, text=True, shell=True
                )
                # Parse Windows ipconfig output
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if interface in line:
                        # Look for IPv4 address in next few lines
                        for j in range(i, min(i+5, len(lines))):
                            if 'IPv4' in lines[j]:
                                ip = lines[j].split(':')[1].strip()
                                return ip
                                
        except Exception as e:
            self.logger.error(f"Failed to get IP for {interface}: {e}")
            
        return None
    
    def rotate_ip(self, hotspot: Hotspot) -> bool:
        """Force IP rotation on hotspot"""
        self.logger.info(f"Rotating IP on {hotspot.name}")
        
        try:
            if platform.system() == 'Linux':
                # Method 1: Toggle airplane mode (requires nmcli)
                subprocess.run(['nmcli', 'radio', 'all', 'off'], check=True)
                time.sleep(5)
                subprocess.run(['nmcli', 'radio', 'all', 'on'], check=True)
                time.sleep(10)
                
                # Method 2: Restart network interface
                subprocess.run(['sudo', 'ifconfig', hotspot.interface, 'down'], check=True)
                time.sleep(2)
                subprocess.run(['sudo', 'ifconfig', hotspot.interface, 'up'], check=True)
                time.sleep(5)
                
            elif platform.system() == 'Windows':
                # Disable and re-enable adapter
                subprocess.run(
                    f'netsh interface set interface "{hotspot.interface}" disabled',
                    shell=True, check=True
                )
                time.sleep(5)
                subprocess.run(
                    f'netsh interface set interface "{hotspot.interface}" enabled',
                    shell=True, check=True
                )
                time.sleep(10)
            
            # Verify new IP
            old_ip = hotspot.current_ip
            new_ip = self.get_current_ip(hotspot.interface)
            
            if new_ip and new_ip != old_ip:
                hotspot.current_ip = new_ip
                hotspot.last_rotation = time.time()
                self.logger.info(f"IP rotated: {old_ip} -> {new_ip}")
                return True
            else:
                self.logger.warning("IP rotation failed - same IP")
                return False
                
        except Exception as e:
            self.logger.error(f"IP rotation failed: {e}")
            return False
    
    def switch_hotspot(self, target_hotspot_id: Optional[str] = None) -> bool:
        """Switch to a different hotspot"""
        if target_hotspot_id:
            if target_hotspot_id not in self.hotspots:
                return False
            target = self.hotspots[target_hotspot_id]
        else:
            # Choose random available hotspot
            available = [h for h in self.hotspots.values() if not h.blocked]
            if not available:
                self.logger.error("No available hotspots!")
                return False
            target = random.choice(available)
        
        self.logger.info(f"Switching to hotspot: {target.name}")
        
        try:
            if platform.system() == 'Linux':
                # Disconnect current
                if self.active_hotspot:
                    subprocess.run(
                        ['nmcli', 'device', 'disconnect', self.active_hotspot.interface],
                        check=True
                    )
                
                # Connect to new hotspot
                # This assumes hotspot SSID is configured in NetworkManager
                subprocess.run(
                    ['nmcli', 'connection', 'up', target.name],
                    check=True
                )
                
            elif platform.system() == 'Windows':
                # Windows network switching
                subprocess.run(
                    f'netsh wlan connect name="{target.name}"',
                    shell=True, check=True
                )
            
            time.sleep(10)  # Wait for connection
            
            # Update current IP
            target.current_ip = self.get_current_ip(target.interface)
            self.active_hotspot = target
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch hotspot: {e}")
            return False
    
    def get_healthiest_hotspot(self) -> Optional[Hotspot]:
        """Get the best hotspot based on usage and last rotation"""
        available = [h for h in self.hotspots.values() if not h.blocked]
        
        if not available:
            return None
        
        # Sort by usage count and last rotation time
        sorted_hotspots = sorted(
            available,
            key=lambda h: (h.usage_count, -h.last_rotation)
        )
        
        return sorted_hotspots[0]

# utils/reddit_auth_manager.py
"""
Reddit API Authentication Manager
Handles multiple Reddit accounts with proper rate limiting
"""

import praw
import json
import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class RedditAccount:
    """Reddit account credentials"""
    username: str
    password: str
    client_id: str
    client_secret: str
    user_agent: str
    last_used: float = 0
    request_count: int = 0
    rate_limit_reset: float = 0
    suspended: bool = False
    
class RedditAuthManager:
    """
    Manages multiple Reddit accounts for API access
    Implements proper rate limiting per OAuth guidelines
    """
    
    def __init__(self, accounts_file: str = 'config/reddit_accounts.json'):
        self.logger = logging.getLogger('reddit_auth')
        self.accounts = self._load_accounts(accounts_file)
        self.instances = {}  # PRAW instances
        self.rate_limits = defaultdict(dict)
        
        # Reddit API limits
        self.REQUESTS_PER_MINUTE = 60
        self.REQUESTS_PER_10_MINUTES = 600
        self.OAUTH_REQUESTS_PER_MINUTE = 60
        
    def _load_accounts(self, accounts_file: str) -> Dict[str, RedditAccount]:
        """Load Reddit accounts from file"""
        accounts = {}
        
        try:
            with open(accounts_file, 'r') as f:
                data = json.load(f)
                
            for acc in data['accounts']:
                account = RedditAccount(**acc)
                accounts[account.username] = account
                
            self.logger.info(f"Loaded {len(accounts)} Reddit accounts")
            
        except FileNotFoundError:
            self.logger.warning(f"No accounts file found at {accounts_file}")
            # Create example file
            example = {
                "accounts": [
                    {
                        "username": "your_username",
                        "password": "your_password",
                        "client_id": "your_client_id",
                        "client_secret": "your_client_secret",
                        "user_agent": "ScrapeHive/1.0 by /u/your_username"
                    }
                ]
            }
            
            with open(accounts_file, 'w') as f:
                json.dump(example, f, indent=2)
                
            self.logger.info(f"Created example accounts file at {accounts_file}")
            
        return accounts
    
    def get_reddit_instance(self, force_account: Optional[str] = None) -> Optional[praw.Reddit]:
        """Get authenticated Reddit instance with rate limiting"""
        
        # Choose account
        if force_account and force_account in self.accounts:
            account = self.accounts[force_account]
        else:
            account = self._choose_best_account()
            
        if not account:
            self.logger.error("No available Reddit accounts")
            return None
        
        # Check rate limits
        if not self._check_rate_limit(account.username):
            self.logger.warning(f"Rate limit hit for {account.username}")
            return None
        
        # Get or create PRAW instance
        if account.username not in self.instances:
            try:
                reddit = praw.Reddit(
                    client_id=account.client_id,
                    client_secret=account.client_secret,
                    username=account.username,
                    password=account.password,
                    user_agent=account.user_agent
                )
                
                # Verify authentication
                reddit.user.me()
                
                self.instances[account.username] = reddit
                self.logger.info(f"Authenticated Reddit account: {account.username}")
                
            except Exception as e:
                self.logger.error(f"Failed to authenticate {account.username}: {e}")
                account.suspended = True
                return None
        
        # Update usage
        account.last_used = time.time()
        account.request_count += 1
        
        return self.instances[account.username]
    
    def _choose_best_account(self) -> Optional[RedditAccount]:
        """Choose the best account based on usage and rate limits"""
        available = [
            acc for acc in self.accounts.values()
            if not acc.suspended and self._check_rate_limit(acc.username)
        ]
        
        if not available:
            return None
        
        # Sort by least recently used
        available.sort(key=lambda x: x.last_used)
        
        return available[0]
    
    def _check_rate_limit(self, username: str) -> bool:
        """Check if account is within rate limits"""
        limits = self.rate_limits[username]
        now = time.time()
        
        # Reset counters if needed
        if 'minute_reset' not in limits or now > limits['minute_reset']:
            limits['minute_requests'] = 0
            limits['minute_reset'] = now + 60
            
        if 'ten_minute_reset' not in limits or now > limits['ten_minute_reset']:
            limits['ten_minute_requests'] = 0
            limits['ten_minute_reset'] = now + 600
        
        # Check limits
        if limits.get('minute_requests', 0) >= self.REQUESTS_PER_MINUTE:
            return False
            
        if limits.get('ten_minute_requests', 0) >= self.REQUESTS_PER_10_MINUTES:
            return False
        
        # Update counters
        limits['minute_requests'] = limits.get('minute_requests', 0) + 1
        limits['ten_minute_requests'] = limits.get('ten_minute_requests', 0) + 1
        
        return True
    
    def handle_rate_limit_response(self, username: str, response_headers: Dict):
        """Update rate limits based on Reddit API response headers"""
        if 'x-ratelimit-remaining' in response_headers:
            remaining = int(response_headers['x-ratelimit-remaining'])
            reset_time = float(response_headers.get('x-ratelimit-reset', 0))
            
            self.rate_limits[username]['remaining'] = remaining
            self.rate_limits[username]['reset'] = reset_time
            
            if remaining < 10:
                self.logger.warning(f"Low rate limit for {username}: {remaining} remaining")

# utils/ip_manager.py
"""
Comprehensive IP Management System
Coordinates proxies, VPNs, and hotspots
"""

import random
import time
import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .hotspot_manager import HotspotManager
from .proxy_providers import ProxyProvider

class IPType(Enum):
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"
    MOBILE = "mobile"
    VPN = "vpn"
    HOTSPOT = "hotspot"

@dataclass
class IPConfig:
    """IP configuration for requests"""
    ip_type: IPType
    address: str
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    provider: Optional[str] = None
    last_used: float = 0
    usage_count: int = 0
    blocked_on: List[str] = None  # Platforms where this IP is blocked
    
    def to_proxy_dict(self) -> Dict:
        """Convert to requests proxy format"""
        if self.username and self.password:
            proxy_url = f"http://{self.username}:{self.password}@{self.address}:{self.port}"
        else:
            proxy_url = f"http://{self.address}:{self.port}"
            
        return {
            'http': proxy_url,
            'https': proxy_url
        }

class IPManager:
    """
    Master IP management system
    Coordinates all IP sources for maximum anonymity
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ip_manager')
        
        # Initialize components
        self.hotspot_manager = HotspotManager()
        self.proxy_provider = ProxyProvider()
        
        # IP pools
        self.ip_pools = {
            IPType.RESIDENTIAL: [],
            IPType.DATACENTER: [],
            IPType.MOBILE: [],
            IPType.VPN: [],
            IPType.HOTSPOT: []
        }
        
        # Platform requirements
        self.platform_requirements = {
            'reddit': [IPType.RESIDENTIAL, IPType.DATACENTER, IPType.HOTSPOT],
            'twitter': [IPType.RESIDENTIAL, IPType.MOBILE],
            'instagram': [IPType.MOBILE, IPType.RESIDENTIAL],
            'tiktok': [IPType.MOBILE, IPType.RESIDENTIAL],
            'facebook': [IPType.RESIDENTIAL]
        }
        
        # Load IPs
        self._initialize_ip_pools()
        
    def _initialize_ip_pools(self):
        """Initialize all IP sources"""
        
        # Load proxies from providers
        self._load_proxies()
        
        # Add hotspots to mobile pool
        for hotspot in self.hotspot_manager.hotspots.values():
            if hotspot.current_ip:
                ip_config = IPConfig(
                    ip_type=IPType.HOTSPOT,
                    address=hotspot.current_ip,
                    provider='hotspot',
                    blocked_on=[]
                )
                self.ip_pools[IPType.HOTSPOT].append(ip_config)
                self.ip_pools[IPType.MOBILE].append(ip_config)  # Also counts as mobile
    
    def _load_proxies(self):
        """Load proxies from configured providers"""
        
        # Example proxy configurations
        proxy_configs = [
            # Residential proxies
            {
                'type': IPType.RESIDENTIAL,
                'provider': 'brightdata',
                'endpoint': 'http://zproxy.lum-superproxy.io:22225',
                'username': 'user-sp-{session}',
                'password': 'pass'
            },
            # Datacenter proxies
            {
                'type': IPType.DATACENTER,
                'provider': 'smartproxy',
                'endpoint': 'gate.smartproxy.com:7000',
                'username': 'user',
                'password': 'pass'
            }
        ]
        
        for config in proxy_configs:
            # In production, fetch proxy list from provider API
            # For now, create example entries
            for i in range(10):
                ip_config = IPConfig(
                    ip_type=config['type'],
                    address=config['endpoint'].split(':')[0],
                    port=int(config['endpoint'].split(':')[1]),
                    username=config['username'].replace('{session}', str(int(time.time() * 1000))),
                    password=config['password'],
                    provider=config['provider'],
                    blocked_on=[]
                )
                self.ip_pools[config['type']].append(ip_config)
    
    def get_ip_for_platform(self, platform: str, exclude_ips: List[str] = None) -> Optional[IPConfig]:
        """Get best IP for specific platform"""
        
        # Get platform requirements
        allowed_types = self.platform_requirements.get(platform, list(IPType))
        
        # Filter available IPs
        available_ips = []
        for ip_type in allowed_types:
            for ip_config in self.ip_pools[ip_type]:
                # Skip blocked IPs
                if platform in (ip_config.blocked_on or []):
                    continue
                    
                # Skip excluded IPs
                if exclude_ips and ip_config.address in exclude_ips:
                    continue
                    
                available_ips.append(ip_config)
        
        if not available_ips:
            self.logger.error(f"No available IPs for {platform}")
            return None
        
        # Sort by usage and last used time
        available_ips.sort(key=lambda x: (x.usage_count, x.last_used))
        
        # Choose best IP
        chosen_ip = available_ips[0]
        
        # For mobile platforms, prefer hotspot if available
        if platform in ['instagram', 'tiktok'] and self.hotspot_manager.hotspots:
            hotspot_ips = [ip for ip in available_ips if ip.ip_type == IPType.HOTSPOT]
            if hotspot_ips:
                chosen_ip = hotspot_ips[0]
        
        # Update usage
        chosen_ip.last_used = time.time()
        chosen_ip.usage_count += 1
        
        self.logger.info(f"Using {chosen_ip.ip_type.value} IP for {platform}: {chosen_ip.address}")
        
        return chosen_ip
    
    def rotate_ip(self, current_ip: IPConfig, reason: str = "scheduled") -> Optional[IPConfig]:
        """Rotate to a new IP"""
        self.logger.info(f"Rotating IP due to: {reason}")
        
        # If it's a hotspot, try to rotate the hotspot IP first
        if current_ip.ip_type == IPType.HOTSPOT:
            hotspot = next(
                (h for h in self.hotspot_manager.hotspots.values() 
                 if h.current_ip == current_ip.address),
                None
            )
            
            if hotspot and self.hotspot_manager.rotate_ip(hotspot):
                # Update IP config
                current_ip.address = hotspot.current_ip
                return current_ip
        
        # Otherwise, get a new IP
        platform = "general"  # Could be passed as parameter
        return self.get_ip_for_platform(platform, exclude_ips=[current_ip.address])
    
    def mark_ip_blocked(self, ip_config: IPConfig, platform: str):
        """Mark an IP as blocked on a platform"""
        if ip_config.blocked_on is None:
            ip_config.blocked_on = []
            
        if platform not in ip_config.blocked_on:
            ip_config.blocked_on.append(platform)
            self.logger.warning(f"IP {ip_config.address} blocked on {platform}")
    
    def test_ip(self, ip_config: IPConfig) -> Tuple[bool, str]:
        """Test if IP is working"""
        try:
            # Test with httpbin
            response = requests.get(
                'http://httpbin.org/ip',
                proxies=ip_config.to_proxy_dict() if ip_config.port else None,
                timeout=10
            )
            
            detected_ip = response.json()['origin']
            
            # For hotspots, the detected IP should match
            if ip_config.ip_type == IPType.HOTSPOT:
                return detected_ip == ip_config.address, detected_ip
            else:
                # For proxies, just check that it's different from our real IP
                return response.status_code == 200, detected_ip
                
        except Exception as e:
            self.logger.error(f"IP test failed for {ip_config.address}: {e}")
            return False, str(e)
    
    def get_ip_stats(self) -> Dict:
        """Get statistics about IP pools"""
        stats = {
            'total_ips': 0,
            'by_type': {},
            'blocked_count': 0,
            'hotspots_available': len(self.hotspot_manager.hotspots)
        }
        
        for ip_type, ips in self.ip_pools.items():
            active = [ip for ip in ips if not ip.blocked_on]
            stats['by_type'][ip_type.value] = {
                'total': len(ips),
                'active': len(active),
                'blocked': len(ips) - len(active)
            }
            stats['total_ips'] += len(ips)
            stats['blocked_count'] += len(ips) - len(active)
        
        return stats