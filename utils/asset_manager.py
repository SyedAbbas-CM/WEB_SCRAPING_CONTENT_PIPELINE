# utils/asset_manager.py
"""
Comprehensive asset management system for video generation
Handles background videos, music, sound effects, fonts, and more
"""

import os
import json
import hashlib
import requests
import random
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
from pydub import AudioSegment

@dataclass
class Asset:
    """Represents a media asset"""
    id: str
    type: str  # video, audio, image, font
    category: str  # background, music, sfx, etc.
    filename: str
    path: str
    metadata: Dict
    tags: List[str]
    duration: Optional[float] = None
    size: int = 0
    hash: str = ""
    created_at: float = 0
    last_used: float = 0
    usage_count: int = 0
    license: str = "unknown"
    source: str = "unknown"

class AssetManager:
    """
    Manages all media assets for video generation
    Features:
    - Asset organization and categorization
    - Automatic downloading from various sources
    - License tracking
    - Usage analytics
    - Smart caching
    """
    
    def __init__(self, assets_dir: str = 'assets', db_path: str = 'assets.db'):
        self.assets_dir = assets_dir
        self.db_path = db_path
        self.logger = logging.getLogger('asset_manager')
        
        # Create directory structure
        self._create_directories()
        
        # Initialize database
        self._init_database()
        
        # Asset sources configuration
        self.sources = {
            'pixabay': {
                'api_key': os.getenv('PIXABAY_API_KEY'),
                'base_url': 'https://pixabay.com/api/'
            },
            'pexels': {
                'api_key': os.getenv('PEXELS_API_KEY'),
                'base_url': 'https://api.pexels.com/v1/'
            },
            'freesound': {
                'api_key': os.getenv('FREESOUND_API_KEY'),
                'base_url': 'https://freesound.org/apiv2/'
            }
        }
        
        # Download executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Popular background video categories
        self.background_presets = {
            'minecraft_parkour': {
                'search_terms': ['minecraft parkour', 'minecraft jumping'],
                'duration_range': (60, 300),
                'resolution': '1920x1080'
            },
            'subway_surfers': {
                'search_terms': ['subway surfers gameplay', 'subway surfers run'],
                'duration_range': (60, 300),
                'resolution': '1080x1920'
            },
            'satisfying': {
                'search_terms': ['satisfying video', 'oddly satisfying', 'soap cutting'],
                'duration_range': (30, 180),
                'resolution': '1080x1920'
            },
            'nature_scenery': {
                'search_terms': ['nature 4k', 'scenic landscape', 'drone footage nature'],
                'duration_range': (60, 300),
                'resolution': '1920x1080'
            },
            'city_lights': {
                'search_terms': ['city lights timelapse', 'urban nightscape', 'city traffic'],
                'duration_range': (60, 300),
                'resolution': '1920x1080'
            }
        }
        
        # Music categories
        self.music_categories = {
            'upbeat': ['energetic', 'happy', 'uplifting', 'motivational'],
            'dramatic': ['epic', 'cinematic', 'intense', 'emotional'],
            'chill': ['relaxing', 'ambient', 'calm', 'peaceful'],
            'funny': ['comedy', 'quirky', 'playful', 'cartoon'],
            'suspense': ['mysterious', 'dark', 'tension', 'thriller']
        }
    
    def _create_directories(self):
        """Create asset directory structure"""
        directories = [
            'backgrounds/gaming',
            'backgrounds/nature',
            'backgrounds/abstract',
            'backgrounds/lifestyle',
            'music/upbeat',
            'music/dramatic',
            'music/chill',
            'music/funny',
            'sfx/transitions',
            'sfx/impacts',
            'sfx/whoosh',
            'sfx/ui',
            'fonts',
            'overlays',
            'temp'
        ]
        
        for directory in directories:
            path = os.path.join(self.assets_dir, directory)
            os.makedirs(path, exist_ok=True)
    
    def _init_database(self):
        """Initialize asset database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata TEXT,
                tags TEXT,
                duration REAL,
                size INTEGER,
                hash TEXT,
                created_at REAL,
                last_used REAL,
                usage_count INTEGER DEFAULT 0,
                license TEXT,
                source TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT,
                used_at REAL,
                video_id TEXT,
                context TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_type_category ON assets(type, category);
            CREATE INDEX IF NOT EXISTS idx_tags ON assets(tags);
            CREATE INDEX IF NOT EXISTS idx_usage ON assets(usage_count);
        ''')
        
        conn.commit()
        conn.close()
    
    def get_background_video(self, category: str, style: Optional[str] = None,
                           min_duration: float = 60) -> Optional[Asset]:
        """
        Get a background video for the specified category
        
        Args:
            category: Category of background (gaming, nature, etc.)
            style: Optional style preference
            min_duration: Minimum duration in seconds
        
        Returns:
            Asset object or None if not found
        """
        # Check local assets first
        asset = self._get_local_asset('video', f'backgrounds/{category}', min_duration)
        
        if asset:
            self._record_usage(asset.id)
            return asset
        
        # Download if not available
        self.logger.info(f"No suitable background found, downloading for category: {category}")
        
        if category in self.background_presets:
            preset = self.background_presets[category]
            search_term = random.choice(preset['search_terms'])
            
            # Try to download from various sources
            asset = self._download_background_video(
                search_term,
                category,
                preset['duration_range'],
                preset['resolution']
            )
            
            if asset:
                self._record_usage(asset.id)
                return asset
        
        # Fallback to any video in the category
        return self._get_any_asset('video', 'backgrounds')
    
    def get_background_music(self, mood: str, duration: float = 60) -> Optional[Asset]:
        """
        Get background music matching the specified mood
        
        Args:
            mood: Mood/style of music (upbeat, dramatic, etc.)
            duration: Desired duration in seconds
        
        Returns:
            Asset object or None if not found
        """
        # Map mood to category
        category = f'music/{mood}'
        
        # Check local assets
        asset = self._get_local_asset('audio', category, duration * 0.8)  # Allow slightly shorter
        
        if asset:
            self._record_usage(asset.id)
            return asset
        
        # Download if needed
        self.logger.info(f"Downloading music for mood: {mood}")
        
        tags = self.music_categories.get(mood, [mood])
        search_term = random.choice(tags) + ' music'
        
        asset = self._download_music(search_term, mood, duration)
        
        if asset:
            self._record_usage(asset.id)
            return asset
        
        return None
    
    def get_sound_effect(self, effect_type: str) -> Optional[Asset]:
        """Get a sound effect"""
        category = f'sfx/{effect_type}'
        
        asset = self._get_local_asset('audio', category)
        
        if not asset:
            # Download sound effect
            asset = self._download_sound_effect(effect_type)
        
        if asset:
            self._record_usage(asset.id)
            
        return asset
    
    def _get_local_asset(self, asset_type: str, category: str, 
                        min_duration: Optional[float] = None) -> Optional[Asset]:
        """Get asset from local database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM assets 
            WHERE type = ? AND category = ?
        '''
        params = [asset_type, category]
        
        if min_duration:
            query += ' AND duration >= ?'
            params.append(min_duration)
        
        # Order by least recently used to distribute usage
        query += ' ORDER BY last_used ASC, usage_count ASC LIMIT 1'
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return self._row_to_asset(row)
        
        return None
    
    def _get_any_asset(self, asset_type: str, category_prefix: str) -> Optional[Asset]:
        """Get any asset matching type and category prefix"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM assets 
            WHERE type = ? AND category LIKE ?
            ORDER BY RANDOM() LIMIT 1
        ''', (asset_type, f'{category_prefix}%'))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_asset(row)
        
        return None
    
    def _download_background_video(self, search_term: str, category: str,
                                 duration_range: Tuple[int, int],
                                 resolution: str) -> Optional[Asset]:
        """Download background video from various sources"""
        
        # Try YouTube first (using yt-dlp)
        try:
            ydl_opts = {
                'format': f'best[height<={resolution.split("x")[1]}]',
                'outtmpl': os.path.join(self.assets_dir, 'temp', '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search for videos
                search_url = f"ytsearch5:{search_term} no copyright {duration_range[0]}-{duration_range[1]} seconds"
                result = ydl.extract_info(search_url, download=False)
                
                # Find suitable video
                for entry in result.get('entries', []):
                    duration = entry.get('duration', 0)
                    if duration_range[0] <= duration <= duration_range[1]:
                        # Download the video
                        ydl_opts['extract_flat'] = False
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                            info = ydl_download.extract_info(entry['url'], download=True)
                            
                            # Process downloaded file
                            filename = ydl_download.prepare_filename(info)
                            return self._process_downloaded_video(
                                filename, category, search_term, 'youtube'
                            )
        
        except Exception as e:
            self.logger.error(f"YouTube download failed: {e}")
        
        # Try Pixabay
        if self.sources['pixabay']['api_key']:
            try:
                return self._download_from_pixabay(search_term, category, 'video')
            except Exception as e:
                self.logger.error(f"Pixabay download failed: {e}")
        
        # Try Pexels
        if self.sources['pexels']['api_key']:
            try:
                return self._download_from_pexels(search_term, category, 'video')
            except Exception as e:
                self.logger.error(f"Pexels download failed: {e}")
        
        return None
    
    def _download_music(self, search_term: str, mood: str, duration: float) -> Optional[Asset]:
        """Download background music"""
        
        # Try Freesound
        if self.sources['freesound']['api_key']:
            try:
                return self._download_from_freesound(search_term, f'music/{mood}')
            except Exception as e:
                self.logger.error(f"Freesound download failed: {e}")
        
        # Try Pixabay Audio
        if self.sources['pixabay']['api_key']:
            try:
                return self._download_from_pixabay(search_term + ' music', f'music/{mood}', 'audio')
            except Exception as e:
                self.logger.error(f"Pixabay audio download failed: {e}")
        
        # Generate procedural music as fallback
        return self._generate_procedural_music(mood, duration)
    
    def _download_sound_effect(self, effect_type: str) -> Optional[Asset]:
        """Download sound effect"""
        search_term = effect_type.replace('_', ' ') + ' sound effect'
        
        # Try Freesound first
        if self.sources['freesound']['api_key']:
            try:
                return self._download_from_freesound(search_term, f'sfx/{effect_type}')
            except Exception as e:
                self.logger.error(f"Freesound SFX download failed: {e}")
        
        # Generate if download fails
        return self._generate_sound_effect(effect_type)
    
    def _download_from_pixabay(self, search_term: str, category: str, 
                             media_type: str = 'video') -> Optional[Asset]:
        """Download from Pixabay"""
        api_key = self.sources['pixabay']['api_key']
        base_url = self.sources['pixabay']['base_url']
        
        # API request
        params = {
            'key': api_key,
            'q': search_term,
            'media_type': media_type,
            'per_page': 5,
            'safesearch': 'true'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['hits']:
                # Choose random result
                hit = random.choice(data['hits'])
                
                # Download URL
                if media_type == 'video':
                    