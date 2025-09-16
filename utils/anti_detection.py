# utils/anti_detection.py
"""
Advanced anti-detection measures for web scraping
Implements browser fingerprinting, behavior randomization, and detection evasion
"""

import random
import time
import json
from typing import Dict, List, Optional, Tuple
from fake_useragent import UserAgent
import logging

class AntiDetection:
    """
    Comprehensive anti-detection system including:
    - Dynamic header rotation
    - Browser fingerprint spoofing
    - Human-like behavior simulation
    - Canvas fingerprint randomization
    - WebRTC leak prevention
    """
    
    def __init__(self):
        self.ua = UserAgent()
        self.logger = logging.getLogger('anti_detection')
        
        # Browser fingerprints database
        self.fingerprints = self._load_fingerprints()
        
        # Behavioral patterns
        self.mouse_patterns = self._load_mouse_patterns()
        self.typing_patterns = self._load_typing_patterns()
        
        # Detection evasion techniques
        self.evasion_scripts = self._load_evasion_scripts()
        
    def get_headers(self, platform: str, browser_profile: Optional[str] = None) -> Dict:
        """Get platform-specific headers with anti-detection measures"""
        
        # Compliance: optionally disable stealth headers via env
        import os
        if os.getenv('COMPLIANCE_MODE', '').lower() in ['api_only', 'compliant']:
            # Minimal realistic headers without platform spoofing
            return {
                'User-Agent': self._get_chrome_ua(),
                'Accept': 'application/json,text/html;q=0.8,*/*;q=0.5',
                'Accept-Language': self._get_random_language(),
                'Connection': 'keep-alive'
            }
        
        # Select browser profile
        if not browser_profile:
            browser_profile = random.choice(['chrome', 'firefox', 'safari', 'edge'])
        
        # Get base headers for browser
        headers = self._get_browser_headers(browser_profile)
        
        # Add platform-specific headers
        platform_headers = self._get_platform_headers(platform)
        headers.update(platform_headers)
        
        # Add random variations
        headers = self._add_header_variations(headers)
        
        return headers
    
    def _get_browser_headers(self, browser: str) -> Dict:
        """Get browser-specific headers"""
        
        if browser == 'chrome':
            return {
                'User-Agent': self._get_chrome_ua(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Language': self._get_random_language(),
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
        elif browser == 'firefox':
            return {
                'User-Agent': self._get_firefox_ua(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': self._get_random_language(),
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'TE': 'trailers'
            }
            
        elif browser == 'safari':
            return {
                'User-Agent': self._get_safari_ua(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': self._get_random_language(),
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
        else:  # edge
            return {
                'User-Agent': self._get_edge_ua(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'Accept-Language': self._get_random_language(),
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
    
    def _get_platform_headers(self, platform: str) -> Dict:
        """Get platform-specific headers"""
        
        if platform == 'instagram':
            return {
                'X-IG-App-ID': '936619743392459',
                'X-IG-WWW-Claim': '0',
                'X-Requested-With': 'XMLHttpRequest',
                'X-ASBD-ID': '198387',
                'X-CSRFToken': self._generate_csrf_token(),
                'X-Instagram-AJAX': '1006670816'
            }
            
        elif platform == 'twitter':
            # Do not hardcode tokens; expect env-provided when needed
            import os
            bearer = os.getenv('TWITTER_BEARER_TOKEN')
            headers = {}
            if bearer:
                headers['Authorization'] = f'Bearer {bearer}'
            return headers
            
        elif platform == 'tiktok':
            return {
                'X-Secsdk-Csrf-Token': self._generate_csrf_token(),
                'X-SS-TC': '0',
                'tt-webid': self._generate_tiktok_webid()
            }
            
        elif platform == 'reddit':
            return {
                'X-Reddit-Session': self._generate_reddit_session()
            }
            
        return {}
    
    def _get_chrome_ua(self) -> str:
        """Generate realistic Chrome user agent"""
        chrome_versions = [
            '120.0.0.0', '119.0.0.0', '118.0.5993.89', '117.0.5938.150'
        ]
        windows_versions = [
            'Windows NT 10.0; Win64; x64',
            'Windows NT 11.0; Win64; x64'
        ]
        
        version = random.choice(chrome_versions)
        windows = random.choice(windows_versions)
        
        return f'Mozilla/5.0 ({windows}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36'
    
    def _get_firefox_ua(self) -> str:
        """Generate realistic Firefox user agent"""
        firefox_versions = ['121.0', '120.0', '119.0', '118.0']
        version = random.choice(firefox_versions)
        
        return f'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}'
    
    def _get_safari_ua(self) -> str:
        """Generate realistic Safari user agent"""
        safari_versions = [
            '605.1.15', '604.1.38', '603.3.8'
        ]
        webkit_versions = [
            '537.36', '605.1.15', '604.1'
        ]
        
        safari = random.choice(safari_versions)
        webkit = random.choice(webkit_versions)
        
        return f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/{webkit} (KHTML, like Gecko) Version/17.1 Safari/{safari}'
    
    def _get_edge_ua(self) -> str:
        """Generate realistic Edge user agent"""
        edge_versions = ['120.0.2210.77', '119.0.2151.97', '118.0.2088.76']
        version = random.choice(edge_versions)
        
        return f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/{version}'
    
    def _get_random_language(self) -> str:
        """Get random Accept-Language header"""
        languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.9',
            'en-US,en;q=0.8,es;q=0.6',
            'en-CA,en;q=0.9',
            'en-US,en;q=0.9,fr;q=0.8',
            'en-AU,en;q=0.9',
            'en-US,en;q=0.8,de;q=0.6'
        ]
        return random.choice(languages)
    
    def _add_header_variations(self, headers: Dict) -> Dict:
        """Add random variations to headers"""
        
        # Randomly add or remove certain headers
        optional_headers = {
            'Sec-CH-UA': '"Not_A Brand";v="8", "Chromium";v="120"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Purpose': 'prefetch',
            'Pragma': 'no-cache'
        }
        
        for header, value in optional_headers.items():
            if random.random() > 0.5:
                headers[header] = value
        
        # Vary header order (return as OrderedDict to maintain order)
        from collections import OrderedDict
        keys = list(headers.keys())
        random.shuffle(keys)
        
        return OrderedDict((k, headers[k]) for k in keys)
    
    def _generate_csrf_token(self) -> str:
        """Generate realistic CSRF token"""
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        return ''.join(random.choice(chars) for _ in range(32))
    
    def _generate_tiktok_webid(self) -> str:
        """Generate TikTok web ID"""
        return str(random.randint(6800000000000000000, 6999999999999999999))
    
    def _generate_reddit_session(self) -> str:
        """Generate Reddit session ID"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choice(chars) for _ in range(24))
    
    def get_browser_fingerprint(self) -> Dict:
        """Get complete browser fingerprint"""
        return {
            'screen': self._get_screen_fingerprint(),
            'canvas': self._get_canvas_fingerprint(),
            'webgl': self._get_webgl_fingerprint(),
            'audio': self._get_audio_fingerprint(),
            'fonts': self._get_font_fingerprint(),
            'plugins': self._get_plugin_fingerprint(),
            'hardware': self._get_hardware_fingerprint()
        }
    
    def _get_screen_fingerprint(self) -> Dict:
        """Generate screen fingerprint"""
        resolutions = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1600, 900), (1280, 720), (2560, 1440), (3840, 2160)
        ]
        
        width, height = random.choice(resolutions)
        
        return {
            'width': width,
            'height': height,
            'availWidth': width,
            'availHeight': height - random.randint(40, 100),  # Taskbar
            'colorDepth': random.choice([24, 32]),
            'pixelDepth': random.choice([24, 32])
        }
    
    def _get_canvas_fingerprint(self) -> str:
        """Generate canvas fingerprint"""
        # Simulate canvas fingerprint hash
        return ''.join(random.choice('0123456789abcdef') for _ in range(32))
    
    def _get_webgl_fingerprint(self) -> Dict:
        """Generate WebGL fingerprint"""
        vendors = [
            'Intel Inc.', 'NVIDIA Corporation', 'ATI Technologies Inc.',
            'Intel Corporation', 'AMD'
        ]
        
        renderers = [
            'Intel HD Graphics 620', 'NVIDIA GeForce GTX 1050',
            'Intel UHD Graphics 630', 'AMD Radeon Pro 560X',
            'Intel Iris Plus Graphics 640'
        ]
        
        return {
            'vendor': random.choice(vendors),
            'renderer': random.choice(renderers),
            'version': 'WebGL 1.0 (OpenGL ES 2.0 Chromium)'
        }
    
    def _get_audio_fingerprint(self) -> float:
        """Generate audio fingerprint"""
        # Simulate audio context fingerprint
        return round(random.uniform(124.043, 124.049), 6)
    
    def _get_font_fingerprint(self) -> List[str]:
        """Generate font list fingerprint"""
        base_fonts = [
            'Arial', 'Arial Black', 'Arial Narrow', 'Book Antiqua',
            'Calibri', 'Cambria', 'Century', 'Comic Sans MS',
            'Consolas', 'Courier', 'Courier New', 'Georgia',
            'Helvetica', 'Impact', 'Lucida Console', 'Palatino Linotype',
            'Segoe UI', 'Tahoma', 'Times', 'Times New Roman',
            'Trebuchet MS', 'Verdana'
        ]
        
        # Randomly include 15-20 fonts
        num_fonts = random.randint(15, 20)
        return random.sample(base_fonts, num_fonts)
    
    def _get_plugin_fingerprint(self) -> List[Dict]:
        """Generate plugin list"""
        plugins = [
            {
                'name': 'Chrome PDF Plugin',
                'filename': 'internal-pdf-viewer',
                'description': 'Portable Document Format'
            },
            {
                'name': 'Chrome PDF Viewer',
                'filename': 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                'description': ''
            },
            {
                'name': 'Native Client',
                'filename': 'internal-nacl-plugin',
                'description': ''
            }
        ]
        
        # Randomly include 0-3 plugins
        num_plugins = random.randint(0, 3)
        return random.sample(plugins, num_plugins)
    
    def _get_hardware_fingerprint(self) -> Dict:
        """Generate hardware fingerprint"""
        return {
            'cpuCores': random.choice([2, 4, 6, 8, 12, 16]),
            'deviceMemory': random.choice([4, 8, 16, 32]),
            'hardwareConcurrency': random.choice([4, 8, 12, 16]),
            'maxTouchPoints': random.choice([0, 1, 5, 10])
        }
    
    def get_selenium_options(self, browser: str = 'chrome') -> Dict:
        """Get Selenium options for anti-detection"""
        options = {
            'arguments': [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--no-sandbox',
                '--window-size=1920,1080',
                '--start-maximized',
                '--user-agent=' + self._get_chrome_ua()
            ],
            'excludeSwitches': ['enable-automation'],
            'useAutomationExtension': False,
            'prefs': {
                'credentials_enable_service': False,
                'profile.password_manager_enabled': False,
                'profile.default_content_setting_values.notifications': 2,
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': False
            }
        }
        
        return options
    
    def get_puppeteer_options(self) -> Dict:
        """Get Puppeteer options for anti-detection"""
        return {
            'headless': False,
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                '--hide-scrollbars',
                '--mute-audio',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ],
            'defaultViewport': {
                'width': 1920,
                'height': 1080
            }
        }
    
    def generate_mouse_movement(self, start: Tuple[int, int], 
                              end: Tuple[int, int], 
                              duration: float = 1.0) -> List[Tuple[int, int, float]]:
        """Generate human-like mouse movement path"""
        import numpy as np
        
        # Generate control points for bezier curve
        ctrl1_x = start[0] + (end[0] - start[0]) * random.uniform(0.1, 0.3)
        ctrl1_y = start[1] + (end[1] - start[1]) * random.uniform(0.1, 0.3)
        
        ctrl2_x = start[0] + (end[0] - start[0]) * random.uniform(0.7, 0.9)
        ctrl2_y = start[1] + (end[1] - start[1]) * random.uniform(0.7, 0.9)
        
        # Generate bezier curve points
        points = []
        steps = int(duration * 60)  # 60 points per second
        
        for i in range(steps):
            t = i / steps
            
            # Bezier curve formula
            x = ((1-t)**3 * start[0] + 
                 3*(1-t)**2*t * ctrl1_x + 
                 3*(1-t)*t**2 * ctrl2_x + 
                 t**3 * end[0])
            
            y = ((1-t)**3 * start[1] + 
                 3*(1-t)**2*t * ctrl1_y + 
                 3*(1-t)*t**2 * ctrl2_y + 
                 t**3 * end[1])
            
            # Add small random variations
            x += random.gauss(0, 2)
            y += random.gauss(0, 2)
            
            # Calculate timestamp
            timestamp = i * (duration / steps)
            
            points.append((int(x), int(y), timestamp))
        
        return points
    
    def generate_typing_delays(self, text: str) -> List[float]:
        """Generate human-like typing delays"""
        delays = []
        
        for i, char in enumerate(text):
            # Base delay
            if char.isspace():
                delay = random.uniform(0.1, 0.3)
            elif char in '.,!?;:':
                delay = random.uniform(0.2, 0.5)
            else:
                delay = random.uniform(0.05, 0.15)
            
            # Add variations
            if random.random() < 0.1:  # 10% chance of pause
                delay += random.uniform(0.5, 1.5)
            
            if random.random() < 0.05:  # 5% chance of typo/correction
                delay += random.uniform(0.3, 0.8)
            
            delays.append(delay)
        
        return delays
    
    def get_stealth_js_injection(self) -> str:
        """Get JavaScript to inject for stealth mode"""
        return '''
        // Override webdriver detection
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        
        // Override plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
        
        // Override permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Override chrome detection
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {}
        };
        
        // Override console.debug
        const originalDebug = console.debug;
        console.debug = function(...args) {
            if (!args[0]?.includes('DevTools')) {
                return originalDebug.apply(console, args);
            }
        };
        '''
    
    def _load_fingerprints(self) -> List[Dict]:
        """Load browser fingerprint database"""
        # In production, load from file/database
        return []
    
    def _load_mouse_patterns(self) -> List[Dict]:
        """Load mouse movement patterns"""
        return []
    
    def _load_typing_patterns(self) -> List[Dict]:
        """Load typing patterns"""
        return []
    
    def _load_evasion_scripts(self) -> Dict[str, str]:
        """Load platform-specific evasion scripts"""
        return {}
