        params = {
            'aweme_id': video_id,
            'version_code': self.device_info['app_version'].replace('.', ''),
            'app_name': self.device_info['app_name'],
            'channel': self.device_info['channel'],
            'device_platform': 'android',
            'device_type': self.device_info['device_type'],
            'os_version': self.device_info['os_version'],
            'aid': '1233',
            'screen_width': self.device_info['resolution'].split('*')[0],
            'screen_height': self.device_info['resolution'].split('*')[1],
            'dpi': self.device_info['dpi']
        }
        
        headers = {
            'User-Agent': f'com.zhiliaoapp.musically/{self.device_info["app_version"]} (Linux; U; Android {self.device_info["os_version"]}; en_US; {self.device_info["device_type"]}; Build/QP1A.190711.020; Cronet/TTNetVersion:5f9540e5 2021-05-20 QuicVersion:47555d5a 2020-10-15)',
            'X-Gorgon': self.x_gorgon or '0',
            'X-Khronos': self.x_khronos or str(int(time.time()))
        }
        
        response = self.session.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('aweme_list'):
                return self._parse_mobile_api_video(data['aweme_list'][0])
        
        raise Exception(f"Mobile API request failed: {response.status_code}")
    
    def _scrape_video_selenium(self, video_id: str) -> Dict:
        """Scrape video using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        # Try to construct full URL if we only have ID
        if video_id.isdigit():
            url = f"{self.base_url}/@tiktok/video/{video_id}"
        else:
            url = f"{self.base_url}/video/{video_id}"
        
        self.driver.get(url)
        
        # Wait for video to load
        try:
            video_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'video'))
            )
        except TimeoutException:
            raise Exception("Video not found or timeout")
        
        # Extract video data
        video_data = {}
        
        # Video URL
        video_data['video_url'] = video_element.get_attribute('src')
        
        # Thumbnail
        video_data['thumbnail'] = video_element.get_attribute('poster')
        
        # Caption
        try:
            caption_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-video-desc"]')
            video_data['caption'] = caption_elem.text
        except:
            video_data['caption'] = ''
        
        # Author
        try:
            author_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-username"]')
            video_data['author'] = {
                'username': author_elem.text.replace('@', ''),
                'url': author_elem.get_attribute('href')
            }
        except:
            video_data['author'] = {}
        
        # Metrics
        metrics = {}
        
        # Likes
        try:
            likes_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-like-count"]')
            metrics['like_count'] = self._parse_count(likes_elem.text)
        except:
            metrics['like_count'] = 0
        
        # Comments
        try:
            comments_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-comment-count"]')
            metrics['comment_count'] = self._parse_count(comments_elem.text)
        except:
            metrics['comment_count'] = 0
        
        # Shares
        try:
            shares_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="share-count"]')
            metrics['share_count'] = self._parse_count(shares_elem.text)
        except:
            metrics['share_count'] = 0
        
        video_data['metrics'] = metrics
        
        # Music
        try:
            music_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="browse-music"]')
            video_data['music'] = {
                'title': music_elem.text,
                'url': music_elem.get_attribute('href')
            }
        except:
            video_data['music'] = {}
        
        return {
            'platform': 'tiktok',
            'type': 'video',
            'id': video_id,
            'url': self.driver.current_url,
            'data': video_data,
            'scraped_at': time.time(),
            'method': 'selenium'
        }
    
    def _scrape_video_alt(self, video_id: str) -> Dict:
        """Scrape video using alternative services"""
        for service in self.alt_services:
            try:
                if service == 'snaptik.app':
                    return self._scrape_snaptik_video(video_id)
                elif service == 'musicaldown.com':
                    return self._scrape_musicaldown_video(video_id)
                elif service == 'tikmate.online':
                    return self._scrape_tikmate_video(video_id)
            except Exception as e:
                self.logger.warning(f"{service} failed: {e}")
                continue
        
        raise Exception("All alternative services failed")
    
    def _scrape_user_web_api(self, username: str, limit: int) -> Dict:
        """Scrape user profile using web API"""
        self._rate_limit()
        
        # First, get user info
        user_info_url = f"{self.web_api_base}/user/detail/"
        
        params = {
            'uniqueId': username,
            'language': 'en',
            'app_name': 'tiktok_web',
            'device_platform': 'web_pc'
        }
        
        headers = self.anti_detection.get_headers('tiktok')
        headers.update({
            'Referer': f'{self.base_url}/@{username}'
        })
        
        response = self.session.get(user_info_url, params=params, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"User info request failed: {response.status_code}")
        
        user_data = response.json()
        if user_data.get('statusCode') != 0:
            raise Exception("User not found")
        
        user_info = user_data['userInfo']
        user_id = user_info['user']['id']
        
        # Get user videos
        videos = []
        cursor = 0
        
        while len(videos) < limit:
            video_url = f"{self.web_api_base}/post/item_list/"
            
            params = {
                'id': user_id,
                'secUid': user_info['user']['secUid'],
                'count': min(30, limit - len(videos)),
                'cursor': cursor,
                'language': 'en',
                'app_name': 'tiktok_web',
                'device_platform': 'web_pc'
            }
            
            response = self.session.get(video_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 0:
                    for item in data.get('itemList', []):
                        video = self._parse_web_api_video(item)
                        videos.append(video)
                    
                    if data.get('hasMore'):
                        cursor = data.get('cursor', cursor + 30)
                    else:
                        break
                else:
                    break
            else:
                break
            
            self._rate_limit()
        
        return {
            'platform': 'tiktok',
            'type': 'user',
            'username': username,
            'user_info': {
                'id': user_info['user']['id'],
                'username': user_info['user']['uniqueId'],
                'nickname': user_info['user']['nickname'],
                'avatar': user_info['user']['avatarLarger'],
                'signature': user_info['user']['signature'],
                'verified': user_info['user']['verified'],
                'follower_count': user_info['stats']['followerCount'],
                'following_count': user_info['stats']['followingCount'],
                'video_count': user_info['stats']['videoCount'],
                'like_count': user_info['stats']['heartCount']
            },
            'videos': videos[:limit],
            'count': len(videos),
            'scraped_at': time.time(),
            'method': 'web_api'
        }
    
    def _scrape_user_mobile_api(self, username: str, limit: int) -> Dict:
        """Scrape user using mobile API"""
        # Similar to web API but with mobile endpoints
        pass
    
    def _scrape_user_selenium(self, username: str, limit: int) -> Dict:
        """Scrape user profile using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f"{self.base_url}/@{username}"
        self.driver.get(url)
        
        # Wait for profile to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-e2e="user-avatar"]'))
            )
        except TimeoutException:
            raise Exception("User profile not found")
        
        # Extract user info
        user_info = {}
        
        try:
            user_info['username'] = username
            
            # Nickname
            nickname_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="user-title"]')
            user_info['nickname'] = nickname_elem.text
            
            # Avatar
            avatar_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="user-avatar"] img')
            user_info['avatar'] = avatar_elem.get_attribute('src')
            
            # Bio
            try:
                bio_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="user-bio"]')
                user_info['signature'] = bio_elem.text
            except:
                user_info['signature'] = ''
            
            # Stats
            stats = {}
            stat_items = self.driver.find_elements(By.CSS_SELECTOR, '[data-e2e="user-stat-item"]')
            
            for item in stat_items:
                label = item.find_element(By.CSS_SELECTOR, 'span:last-child').text.lower()
                count = item.find_element(By.CSS_SELECTOR, 'strong').text
                
                if 'following' in label:
                    stats['following_count'] = self._parse_count(count)
                elif 'followers' in label:
                    stats['follower_count'] = self._parse_count(count)
                elif 'likes' in label:
                    stats['like_count'] = self._parse_count(count)
            
            user_info.update(stats)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract user info: {e}")
        
        # Get videos
        videos = []
        seen_ids = set()
        last_height = 0
        no_change_count = 0
        
        while len(videos) < limit and no_change_count < 3:
            # Get all video elements
            video_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-e2e="user-post-item"]')
            
            for element in video_elements:
                if len(videos) >= limit:
                    break
                
                try:
                    # Get video link
                    link = element.find_element(By.TAG_NAME, 'a')
                    video_url = link.get_attribute('href')
                    video_id = self._extract_video_id(video_url)
                    
                    if video_id and video_id not in seen_ids:
                        seen_ids.add(video_id)
                        
                        # Extract basic video data
                        video_data = {
                            'id': video_id,
                            'url': video_url
                        }
                        
                        # Try to get metrics
                        try:
                            views_elem = element.find_element(By.CSS_SELECTOR, 'strong')
                            video_data['view_count'] = self._parse_count(views_elem.text)
                        except:
                            pass
                        
                        videos.append(video_data)
                        
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
            'platform': 'tiktok',
            'type': 'user',
            'username': username,
            'user_info': user_info,
            'videos': videos[:limit],
            'count': len(videos),
            'scraped_at': time.time(),
            'method': 'selenium'
        }
    
    def _scrape_hashtag_web_api(self, hashtag: str, limit: int) -> Dict:
        """Scrape hashtag using web API"""
        self._rate_limit()
        
        # Get hashtag ID
        challenge_url = f"{self.web_api_base}/challenge/detail/"
        
        params = {
            'challengeName': hashtag,
            'language': 'en',
            'app_name': 'tiktok_web',
            'device_platform': 'web_pc'
        }
        
        headers = self.anti_detection.get_headers('tiktok')
        headers.update({
            'Referer': f'{self.base_url}/tag/{hashtag}'
        })
        
        response = self.session.get(challenge_url, params=params, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Hashtag info request failed: {response.status_code}")
        
        hashtag_data = response.json()
        if hashtag_data.get('statusCode') != 0:
            raise Exception("Hashtag not found")
        
        challenge_info = hashtag_data['challengeInfo']
        challenge_id = challenge_info['challenge']['id']
        
        # Get videos
        videos = []
        cursor = 0
        
        while len(videos) < limit:
            video_url = f"{self.web_api_base}/challenge/item_list/"
            
            params = {
                'challengeID': challenge_id,
                'count': min(30, limit - len(videos)),
                'cursor': cursor,
                'language': 'en',
                'app_name': 'tiktok_web',
                'device_platform': 'web_pc'
            }
            
            response = self.session.get(video_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 0:
                    for item in data.get('itemList', []):
                        video = self._parse_web_api_video(item)
                        videos.append(video)
                    
                    if data.get('hasMore'):
                        cursor = data.get('cursor', cursor + 30)
                    else:
                        break
                else:
                    break
            else:
                break
            
            self._rate_limit()
        
        return {
            'platform': 'tiktok',
            'type': 'hashtag',
            'hashtag': hashtag,
            'hashtag_info': {
                'id': challenge_info['challenge']['id'],
                'title': challenge_info['challenge']['title'],
                'description': challenge_info['challenge']['desc'],
                'video_count': challenge_info['stats']['videoCount'],
                'view_count': challenge_info['stats']['viewCount']
            },
            'videos': videos[:limit],
            'count': len(videos),
            'scraped_at': time.time(),
            'method': 'web_api'
        }
    
    def _scrape_hashtag_selenium(self, hashtag: str, limit: int) -> Dict:
        """Scrape hashtag using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        url = f"{self.base_url}/tag/{hashtag}"
        self.driver.get(url)
        
        # Wait for videos to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-e2e="challenge-item"]'))
            )
        except TimeoutException:
            raise Exception("No videos found for hashtag")
        
        # Extract hashtag info
        hashtag_info = {
            'title': hashtag
        }
        
        try:
            # View count
            view_elem = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="challenge-vvcount"]')
            hashtag_info['view_count'] = self._parse_count(view_elem.text.replace('views', ''))
        except:
            pass
        
        # Get videos
        videos = []
        seen_ids = set()
        last_height = 0
        no_change_count = 0
        
        while len(videos) < limit and no_change_count < 3:
            # Get all video elements
            video_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-e2e="challenge-item"]')
            
            for element in video_elements:
                if len(videos) >= limit:
                    break
                
                try:
                    # Get video link
                    link = element.find_element(By.TAG_NAME, 'a')
                    video_url = link.get_attribute('href')
                    video_id = self._extract_video_id(video_url)
                    
                    if video_id and video_id not in seen_ids:
                        seen_ids.add(video_id)
                        
                        video_data = {
                            'id': video_id,
                            'url': video_url
                        }
                        
                        # Try to get caption
                        try:
                            caption_elem = element.find_element(By.CSS_SELECTOR, '[data-e2e="challenge-item-desc"]')
                            video_data['caption'] = caption_elem.text
                        except:
                            pass
                        
                        videos.append(video_data)
                        
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
            'platform': 'tiktok',
            'type': 'hashtag',
            'hashtag': hashtag,
            'hashtag_info': hashtag_info,
            'videos': videos[:limit],
            'count': len(videos),
            'scraped_at': time.time(),
            'method': 'selenium'
        }
    
    def _scrape_trending_web_api(self, params: Dict) -> Dict:
        """Scrape trending videos using web API"""
        self._rate_limit()
        
        url = f"{self.web_api_base}/recommend/item_list/"
        
        api_params = {
            'count': params.get('limit', 30),
            'language': 'en',
            'app_name': 'tiktok_web',
            'device_platform': 'web_pc',
            'region': params.get('region', 'US')
        }
        
        headers = self.anti_detection.get_headers('tiktok')
        headers.update({
            'Referer': self.base_url
        })
        
        response = self.session.get(url, params=api_params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('statusCode') == 0:
                videos = []
                for item in data.get('itemList', []):
                    video = self._parse_web_api_video(item)
                    videos.append(video)
                
                return {
                    'platform': 'tiktok',
                    'type': 'trending',
                    'region': api_params['region'],
                    'videos': videos,
                    'count': len(videos),
                    'scraped_at': time.time(),
                    'method': 'web_api'
                }
        
        raise Exception(f"Trending request failed: {response.status_code}")
    
    def _scrape_trending_selenium(self, params: Dict) -> Dict:
        """Scrape trending using Selenium"""
        if not self.driver:
            self._init_selenium_driver()
        
        self.driver.get(self.base_url)
        
        # Wait for videos to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-e2e="recommend-list-item-container"]'))
            )
        except TimeoutException:
            raise Exception("No trending videos found")
        
        videos = []
        seen_ids = set()
        limit = params.get('limit', 30)
        
        # Scroll and collect videos
        for _ in range(limit):
            try:
                # Get current video
                video_container = self.driver.find_element(By.CSS_SELECTOR, '[data-e2e="recommend-list-item-container"]')
                
                # Extract video data
                video_data = {}
                
                # Video ID from URL
                current_url = self.driver.current_url
                video_id = self._extract_video_id(current_url)
                
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    video_data['id'] = video_id
                    video_data['url'] = current_url
                    
                    # Caption
                    try:
                        caption_elem = video_container.find_element(By.CSS_SELECTOR, '[data-e2e="browse-video-desc"]')
                        video_data['caption'] = caption_elem.text
                    except:
                        pass
                    
                    # Author
                    try:
                        author_elem = video_container.find_element(By.CSS_SELECTOR, '[data-e2e="browse-username"]')
                        video_data['author'] = author_elem.text.replace('@', '')
                    except:
                        pass
                    
                    # Metrics
                    metrics = {}
                    try:
                        likes_elem = video_container.find_element(By.CSS_SELECTOR, '[data-e2e="browse-like-count"]')
                        metrics['like_count'] = self._parse_count(likes_elem.text)
                    except:
                        pass
                    
                    video_data['metrics'] = metrics
                    videos.append(video_data)
                
                # Swipe to next video
                self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_DOWN)
                time.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error extracting video: {e}")
                continue
        
        return {
            'platform': 'tiktok',
            'type': 'trending',
            'videos': videos,
            'count': len(videos),
            'scraped_at': time.time(),
            'method': 'selenium'
        }
    
    def _parse_web_api_video(self, item: Dict) -> Dict:
        """Parse video from web API response"""
        return {
            'id': item.get('id'),
            'url': f"{self.base_url}/@{item.get('author', {}).get('uniqueId')}/video/{item.get('id')}",
            'caption': item.get('desc', ''),
            'created_at': item.get('createTime'),
            'author': {
                'id': item.get('author', {}).get('id'),
                'username': item.get('author', {}).get('uniqueId'),
                'nickname': item.get('author', {}).get('nickname'),
                'avatar': item.get('author', {}).get('avatarThumb'),
                'verified': item.get('author', {}).get('verified')
            },
            'metrics': {
                'like_count': item.get('stats', {}).get('diggCount', 0),
                'comment_count': item.get('stats', {}).get('commentCount', 0),
                'share_count': item.get('stats', {}).get('shareCount', 0),
                'play_count': item.get('stats', {}).get('playCount', 0)
            },
            'video': {
                'duration': item.get('video', {}).get('duration', 0),
                'cover': item.get('video', {}).get('cover'),
                'download_url': item.get('video', {}).get('downloadAddr'),
                'play_url': item.get('video', {}).get('playAddr')
            },
            'music': {
                'id': item.get('music', {}).get('id'),
                'title': item.get('music', {}).get('title'),
                'author': item.get('music', {}).get('authorName'),
                'original': item.get('music', {}).get('original')
            },
            'hashtags': [tag.get('hashtagName') for tag in item.get('textExtra', []) if tag.get('hashtagName')]
        }
    
    def _parse_mobile_api_video(self, item: Dict) -> Dict:
        """Parse video from mobile API response"""
        # Similar structure to web API but with slightly different field names
        return {
            'id': item.get('aweme_id'),
            'caption': item.get('desc', ''),
            'author': {
                'username': item.get('author', {}).get('unique_id'),
                'nickname': item.get('author', {}).get('nickname')
            },
            'metrics': {
                'like_count': item.get('statistics', {}).get('digg_count', 0),
                'comment_count': item.get('statistics', {}).get('comment_count', 0),
                'share_count': item.get('statistics', {}).get('share_count', 0)
            }
        }
    
    def _init_selenium_driver(self):
        """Initialize Selenium driver with anti-detection"""
        options = webdriver.ChromeOptions()
        
        # Anti-detection options
        selenium_options = self.anti_detection.get_selenium_options()
        for arg in selenium_options['arguments']:
            options.add_argument(arg)
        
        # TikTok specific options
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Create driver
        self.driver = webdriver.Chrome(options=options)
        
        # Inject stealth JavaScript
        stealth_js = self.anti_detection.get_stealth_js_injection()
        self.driver.execute_script(stealth_js)
        
        # Additional TikTok specific stealth
        self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
    
    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        # Add random variation
        time.sleep(random.uniform(0.5, 2.0))
        self.last_request_time = time.time()
    
    def _parse_count(self, text: str) -> int:
        """Parse count from text (1.2K -> 1200)"""
        if not text:
            return 0
        
        text = text.strip().upper()
        
        # Remove any non-numeric characters except K, M, B
        text = re.sub(r'[^0-9KMB.]', '', text)
        
        if 'K' in text:
            return int(float(text.replace('K', '')) * 1000)
        elif 'M' in text:
            return int(float(text.replace('M', '')) * 1000000)
        elif 'B' in text:
            return int(float(text.replace('B', '')) * 1000000000)
        else:
            try:
                return int(float(text))
            except:
                return 0
    
    def _generate_signatures(self):
        """Generate X-Gorgon and X-Khronos signatures"""
        # This is a simplified version - real implementation would require
        # reverse engineering TikTok's signature algorithm
        timestamp = str(int(time.time()))
        self.x_khronos = timestamp
        
        # Generate X-Gorgon (simplified - not real algorithm)
        data = f"{timestamp}{self.device_info['device_id']}"
        self.x_gorgon = hashlib.md5(data.encode()).hexdigest()[:8]
    
    # Alternative service methods
    def _scrape_snaptik_video(self, video_id: str) -> Dict:
        """Scrape video using SnapTik"""
        # Implementation for SnapTik service
        pass
    
    def _scrape_musicaldown_video(self, video_id: str) -> Dict:
        """Scrape video using MusicalDown"""
        # Implementation for MusicalDown service
        pass
    
    def _scrape_tikmate_video(self, video_id: str) -> Dict:
        """Scrape video using TikMate"""
        # Implementation for TikMate service
        pass
    
    def __del__(self):
        """Cleanup"""
        if self.driver:
            self.driver.quit()    def _resolve_short_url(self, short_url: str) -> Optional[str]:
        """Resolve short URL to get video ID"""
        try:
            response = self.session.head(short_url, allow_redirects=True, timeout=10)
            final_url = response.url
            return self._extract_video_id(final_url)
        except:
            return None
    
    def _scrape_video_web_api(self, video_id: str) -> Dict:
        """Scrape video using web API"""
        self._rate_limit()
        
        # Build API URL
        url = f"{self.web_api_base}/item/detail/"
        
        params = {
            'itemId': video_id,
            'language': 'en',
            'app_name': 'tiktok_web',
            'device_platform': 'web_pc',
            'region': 'US'
        }
        
        headers = self.anti_detection.get_headers('tiktok')
        headers.update({
            'Referer': f'{self.base_url}/@user/video/{video_id}'
        })
        
        response = self.session.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('statusCode') == 0:
                return self._parse_web_api_video(data['itemInfo']['itemStruct'])
        
        raise Exception(f"Web API request failed: {response.status_code}")
    
    def _scrape_video_mobile_api(self, video_id: str) -> Dict:
        """Scrape video using mobile API"""
        self._rate_limit()
        
        # Generate signatures
        self._generate_signatures()
        
        url = f"{self.mobile_api_base}/aweme/v1/feed/"
        
        params = {    def scrape_hashtag(self, hashtag: str, limit: int = 30) -> Dict:
        """Scrape hashtag videos"""
        hashtag = hashtag.replace('#', '')
        
        methods = [
            self._scrape_hashtag_web_api,
            self._scrape_hashtag_selenium
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
    
    def scrape_trending(self, params: Dict = None) -> Dict:
        """Scrape trending videos"""
        methods = [
            self._scrape_trending_web_api,
            self._scrape_trending_selenium
        ]
        
        for method in methods:
            try:
                result = method(params or {})
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All trending scraping methods failed")
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL"""
        patterns = [
            r'tiktok\.com/@[\w.-]+/video/(\d+)',
            r'tiktok\.com/v/(\d+)',
            r'vm\.tiktok\.com/(\w+)',
            r't\.tiktok\.com/(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                
                # If it's a short URL, we need to resolve it
                if 'vm.tiktok.com' in url or 't.tiktok.com' in url:
                    video_id = self._resolve_short_url(url)
                
                return video_id
        
        # Check if it's already just an ID
        if url.isdigit():
            return url
        
        return None
    
    def _resolve_short_url(self, short_url: str) -> Optional[str]:
        """Resolve# scrapers/tiktok_scraper.py
"""
TikTok Scraper with advanced anti-detection
Supports multiple methods including mobile API simulation
"""

import re
import json
import time
import random
import hashlib
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote, urlencode, urlparse
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from utils.anti_detection import AntiDetection

class TikTokScraper:
    """
    Advanced TikTok scraper supporting:
    - Mobile API endpoints
    - Web scraping with Selenium
    - TikTok Web API
    - Alternative services (SnapTik, etc.)
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f'tiktok_scraper.{node_id}')
        
        # Anti-detection
        self.anti_detection = AntiDetection()
        
        # Session management
        self.session = requests.Session()
        self.driver = None
        
        # TikTok endpoints
        self.base_url = 'https://www.tiktok.com'
        self.mobile_api_base = 'https://api-h2.tiktokv.com'
        self.web_api_base = 'https://www.tiktok.com/api'
        
        # Device and app info for mobile API
        self.device_info = self._generate_device_info()
        
        # Signature generation
        self.x_gorgon = None
        self.x_khronos = None
        
        # Alternative services
        self.alt_services = [
            'snaptik.app',
            'musicaldown.com',
            'tikmate.online'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0
    
    def _generate_device_info(self) -> Dict:
        """Generate realistic device information"""
        devices = [
            {
                'device_id': self._generate_device_id(),
                'device_type': 'SM-G973F',
                'device_brand': 'samsung',
                'os_version': '10',
                'app_version': '19.1.3',
                'app_name': 'musical_ly',
                'channel': 'googleplay',
                'resolution': '1080*2280',
                'dpi': '420'
            },
            {
                'device_id': self._generate_device_id(),
                'device_type': 'iPhone11,8',
                'device_brand': 'Apple',
                'os_version': '14.4',
                'app_version': '19.1.0',
                'app_name': 'musical_ly',
                'channel': 'App Store',
                'resolution': '828*1792',
                'dpi': '326'
            }
        ]
        
        return random.choice(devices)
    
    def _generate_device_id(self) -> str:
        """Generate device ID"""
        return ''.join(random.choices('0123456789', k=19))
    
    def set_proxy(self, proxy: Dict):
        """Set proxy for requests"""
        if proxy:
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['address']}:{proxy['port']}"
            self.session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
    
    def scrape_video(self, url: str) -> Dict:
        """Scrape a single TikTok video"""
        # Extract video ID from URL
        video_id = self._extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid TikTok URL: {url}")
        
        # Try methods in order
        methods = [
            self._scrape_video_web_api,
            self._scrape_video_mobile_api,
            self._scrape_video_selenium,
            self._scrape_video_alt
        ]
        
        for method in methods:
            try:
                result = method(video_id)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All scraping methods failed")
    
    def scrape_user(self, username: str, limit: int = 30) -> Dict:
        """Scrape user profile and videos"""
        username = username.replace('@', '')
        
        methods = [
            self._scrape_user_web_api,
            self._scrape_user_mobile_api,
            self._scrape_user_selenium
        ]
        
        for method in methods:
            try:
                result = method(username, limit)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise Exception("All user scraping methods failed")
    
    def scrape_hashtag(self, hashtag: str, limit: int = 30) -> Dict:
        """Scrape hashtag videos"""