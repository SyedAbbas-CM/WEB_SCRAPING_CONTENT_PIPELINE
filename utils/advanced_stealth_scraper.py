# utils/advanced_stealth_scraper.py
"""
Advanced stealth scraper with sophisticated anti-detection measures
Use only when official APIs are exhausted and with full respect for robots.txt
"""

import time
import random
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import os
import hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc


@dataclass
class BehaviorPattern:
    """Human-like behavior pattern"""
    scroll_behavior: Dict
    click_behavior: Dict
    typing_behavior: Dict
    pause_behavior: Dict


class AdvancedStealthScraper:
    """
    Advanced stealth scraper with human-like behavior simulation
    Only use when APIs are unavailable and robots.txt allows
    """

    def __init__(self, platform: str):
        self.platform = platform
        self.logger = logging.getLogger(f'stealth_scraper.{platform}')

        # Only proceed if compliance mode allows
        compliance_mode = os.getenv('COMPLIANCE_MODE', 'api_first')
        if compliance_mode in ['api_only', 'compliant']:
            raise Exception("Stealth scraping disabled in current compliance mode")

        # Behavior patterns
        self.behavior_patterns = self._load_behavior_patterns()
        self.current_session_id = self._generate_session_id()

        # Driver management
        self.driver = None
        self.session_duration = 0
        self.max_session_duration = random.randint(1800, 3600)  # 30-60 minutes

        # Detection metrics
        self.detection_signals = {
            'captcha_encountered': 0,
            'rate_limited': 0,
            'blocked_requests': 0,
            'suspicious_responses': 0
        }

    def _load_behavior_patterns(self) -> Dict:
        """Load human-like behavior patterns"""
        return {
            'scroll_patterns': {
                'natural': {
                    'speed_range': (100, 300),
                    'pause_probability': 0.3,
                    'pause_duration': (0.5, 2.0),
                    'scroll_back_probability': 0.1
                },
                'focused': {
                    'speed_range': (200, 500),
                    'pause_probability': 0.2,
                    'pause_duration': (0.2, 1.0),
                    'scroll_back_probability': 0.05
                }
            },
            'typing_patterns': {
                'human': {
                    'base_delay': (0.05, 0.15),
                    'word_pause': (0.1, 0.4),
                    'mistake_probability': 0.02,
                    'correction_delay': (0.3, 0.8)
                }
            },
            'click_patterns': {
                'natural': {
                    'pre_click_delay': (0.1, 0.3),
                    'post_click_delay': (0.5, 1.5),
                    'mouse_movement_steps': (15, 25)
                }
            }
        }

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:12]

    def initialize_driver(self, headless: bool = False) -> webdriver.Chrome:
        """Initialize undetected Chrome driver with stealth configurations"""

        try:
            options = uc.ChromeOptions()

            # Basic stealth options
            if headless:
                options.add_argument('--headless')

            # Advanced anti-detection arguments
            stealth_args = [
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-extensions-except=path/to/extension',
                '--disable-plugins-discovery',
                '--disable-plugins',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-popup-blocking',
                '--disable-translate',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-device-discovery-notifications',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--remote-debugging-port=9222'
            ]

            for arg in stealth_args:
                options.add_argument(arg)

            # Randomize window size
            window_sizes = [(1366, 768), (1920, 1080), (1440, 900), (1536, 864)]
            width, height = random.choice(window_sizes)
            options.add_argument(f'--window-size={width},{height}')

            # Disable automation indicators
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)

            # Additional preferences
            prefs = {
                'profile.default_content_setting_values.notifications': 2,
                'profile.default_content_settings.popups': 0,
                'profile.managed_default_content_settings.images': 2,  # Block images for speed
                'profile.password_manager_enabled': False,
                'credentials_enable_service': False
            }
            options.add_experimental_option('prefs', prefs)

            # Initialize driver
            self.driver = uc.Chrome(options=options)

            # Execute stealth scripts
            self._execute_stealth_scripts()

            self.logger.info(f"Stealth driver initialized for {self.platform}")
            return self.driver

        except Exception as e:
            self.logger.error(f"Failed to initialize stealth driver: {e}")
            raise

    def _execute_stealth_scripts(self):
        """Execute JavaScript to further hide automation"""
        stealth_scripts = [
            # Hide webdriver property
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            """,

            # Spoof plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            """,

            # Override permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """,

            # Randomize canvas fingerprint
            """
            const getContext = HTMLCanvasElement.prototype.getContext;
            HTMLCanvasElement.prototype.getContext = function(type) {
                if (type === '2d') {
                    const context = getContext.apply(this, arguments);
                    const originalFillText = context.fillText;
                    context.fillText = function() {
                        // Add subtle randomization
                        const noise = Math.random() * 0.0001;
                        arguments[1] += noise;
                        arguments[2] += noise;
                        return originalFillText.apply(this, arguments);
                    };
                    return context;
                }
                return getContext.apply(this, arguments);
            };
            """
        ]

        for script in stealth_scripts:
            try:
                self.driver.execute_script(script)
            except Exception as e:
                self.logger.warning(f"Failed to execute stealth script: {e}")

    def human_like_navigation(self, url: str, wait_time: Tuple[int, int] = (2, 5)):
        """Navigate to URL with human-like behavior"""

        if not self.driver:
            self.initialize_driver()

        try:
            # Random delay before navigation
            time.sleep(random.uniform(*wait_time))

            # Check if we should rotate session
            if self.session_duration > self.max_session_duration:
                self._rotate_session()

            # Navigate
            self.driver.get(url)
            self.session_duration += random.uniform(1, 3)

            # Wait for page load with human-like patience
            page_load_wait = random.uniform(2, 4)
            time.sleep(page_load_wait)

            # Random mouse movement to simulate human presence
            self._simulate_human_presence()

            # Check for detection signals
            self._check_detection_signals()

        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            raise

    def _simulate_human_presence(self):
        """Simulate human presence on the page"""
        try:
            actions = ActionChains(self.driver)

            # Random mouse movements
            for _ in range(random.randint(1, 3)):
                x_offset = random.randint(-100, 100)
                y_offset = random.randint(-100, 100)
                actions.move_by_offset(x_offset, y_offset)
                actions.pause(random.uniform(0.1, 0.3))

            # Occasional scroll
            if random.random() < 0.6:
                scroll_amount = random.randint(100, 500)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.5))

                # Sometimes scroll back
                if random.random() < 0.2:
                    self.driver.execute_script(f"window.scrollBy(0, -{scroll_amount//2});")
                    time.sleep(random.uniform(0.3, 0.8))

            actions.perform()

        except Exception as e:
            self.logger.warning(f"Failed to simulate human presence: {e}")

    def smart_element_interaction(self, selector: str, action: str = 'click',
                                text_input: str = None, scroll_to: bool = True) -> bool:
        """Interact with elements using human-like behavior"""

        try:
            # Wait for element
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )

            # Scroll to element if needed
            if scroll_to:
                self._human_scroll_to_element(element)

            # Pre-interaction delay
            time.sleep(random.uniform(0.2, 0.6))

            if action == 'click':
                self._human_like_click(element)
            elif action == 'type' and text_input:
                self._human_like_typing(element, text_input)
            elif action == 'hover':
                ActionChains(self.driver).move_to_element(element).perform()
                time.sleep(random.uniform(0.5, 1.0))

            # Post-interaction delay
            time.sleep(random.uniform(0.3, 1.0))

            return True

        except Exception as e:
            self.logger.warning(f"Element interaction failed: {e}")
            return False

    def _human_like_click(self, element):
        """Perform human-like click with mouse movement"""
        # Get element location
        location = element.location_once_scrolled_into_view
        size = element.size

        # Calculate random click point within element
        x_offset = random.randint(-size['width']//4, size['width']//4)
        y_offset = random.randint(-size['height']//4, size['height']//4)

        # Move to element with human-like path
        actions = ActionChains(self.driver)
        actions.move_to_element_with_offset(element, x_offset, y_offset)

        # Add some randomness to movement
        for _ in range(random.randint(1, 3)):
            actions.move_by_offset(
                random.randint(-2, 2),
                random.randint(-2, 2)
            )
            actions.pause(random.uniform(0.01, 0.05))

        # Click with slight delay
        actions.pause(random.uniform(0.05, 0.15))
        actions.click()
        actions.perform()

    def _human_like_typing(self, element, text: str):
        """Type text with human-like characteristics"""
        element.clear()

        for i, char in enumerate(text):
            # Base typing delay
            delay = random.uniform(0.05, 0.15)

            # Add pauses at word boundaries
            if char == ' ':
                delay = random.uniform(0.1, 0.4)

            # Occasional longer pauses (thinking)
            if random.random() < 0.05:
                delay += random.uniform(0.5, 1.5)

            # Simulate occasional mistakes
            if random.random() < 0.02 and i > 0:
                # Type wrong character
                wrong_chars = 'abcdefghijklmnopqrstuvwxyz'
                wrong_char = random.choice(wrong_chars)
                element.send_keys(wrong_char)
                time.sleep(random.uniform(0.1, 0.3))

                # Backspace and correct
                element.send_keys('\b')
                time.sleep(random.uniform(0.2, 0.5))
                element.send_keys(char)
            else:
                element.send_keys(char)

            time.sleep(delay)

    def _human_scroll_to_element(self, element):
        """Scroll to element with human-like behavior"""
        # Get element position
        y_position = element.location['y']
        current_y = self.driver.execute_script("return window.pageYOffset;")

        # Calculate scroll distance
        viewport_height = self.driver.execute_script("return window.innerHeight;")
        target_y = max(0, y_position - viewport_height // 2)
        scroll_distance = target_y - current_y

        # Scroll in steps for natural behavior
        if abs(scroll_distance) > 100:
            steps = random.randint(3, 7)
            step_size = scroll_distance // steps

            for i in range(steps):
                self.driver.execute_script(f"window.scrollBy(0, {step_size});")
                time.sleep(random.uniform(0.1, 0.3))

            # Final adjustment
            remaining = scroll_distance - (step_size * steps)
            if remaining != 0:
                self.driver.execute_script(f"window.scrollBy(0, {remaining});")
        else:
            self.driver.execute_script(f"window.scrollBy(0, {scroll_distance});")

        time.sleep(random.uniform(0.3, 0.8))

    def _check_detection_signals(self):
        """Check for signs of bot detection"""
        page_source = self.driver.page_source.lower()
        current_url = self.driver.current_url.lower()

        # Common detection indicators
        detection_keywords = [
            'captcha', 'recaptcha', 'blocked', 'bot', 'automation',
            'suspicious activity', 'verify you are human', 'access denied',
            'rate limit', '429', 'too many requests'
        ]

        for keyword in detection_keywords:
            if keyword in page_source or keyword in current_url:
                self.detection_signals['suspicious_responses'] += 1
                self.logger.warning(f"Possible detection signal: {keyword}")

                # Implement response strategy
                self._handle_detection_signal(keyword)
                break

    def _handle_detection_signal(self, signal: str):
        """Handle detection signals"""
        if 'captcha' in signal:
            self.detection_signals['captcha_encountered'] += 1
            self.logger.warning("CAPTCHA detected - implementing delay strategy")
            time.sleep(random.uniform(300, 600))  # 5-10 minute delay

        elif 'rate limit' in signal or '429' in signal:
            self.detection_signals['rate_limited'] += 1
            self.logger.warning("Rate limit detected - extending delays")
            time.sleep(random.uniform(600, 1200))  # 10-20 minute delay

        elif 'blocked' in signal or 'access denied' in signal:
            self.detection_signals['blocked_requests'] += 1
            self.logger.error("Access blocked - may need session rotation")
            self._rotate_session()

    def _rotate_session(self):
        """Rotate session to avoid detection"""
        self.logger.info("Rotating session for anti-detection")

        if self.driver:
            self.driver.quit()

        # Wait before creating new session
        time.sleep(random.uniform(60, 180))

        # Reinitialize
        self.current_session_id = self._generate_session_id()
        self.session_duration = 0
        self.max_session_duration = random.randint(1800, 3600)

        self.initialize_driver()

    def extract_page_data(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract data from page using provided selectors"""
        data = {}

        for field, selector in selectors.items():
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                if elements:
                    if len(elements) == 1:
                        data[field] = elements[0].text.strip()
                    else:
                        data[field] = [elem.text.strip() for elem in elements]
                else:
                    data[field] = None

            except Exception as e:
                self.logger.warning(f"Failed to extract {field}: {e}")
                data[field] = None

        return data

    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            **self.detection_signals,
            'session_id': self.current_session_id,
            'session_duration': self.session_duration,
            'max_session_duration': self.max_session_duration
        }

    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None

        self.logger.info("Stealth scraper cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()