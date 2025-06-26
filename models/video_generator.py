# utils/video_generator.py
"""
Advanced video generation module for creating viral short-form content
Supports various styles and platforms (TikTok, YouTube Shorts, Instagram Reels)
"""

import os
import json
import random
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
from moviepy.editor import *
from moviepy.video.fx import *
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
from gtts import gTTS
import requests

@dataclass
class VideoConfig:
    """Video generation configuration"""
    width: int = 1080
    height: int = 1920  # 9:16 aspect ratio for shorts
    fps: int = 30
    duration: int = 60  # seconds
    style: str = 'modern'  # modern, minimal, gaming, meme
    platform: str = 'tiktok'  # tiktok, youtube, instagram
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

class VideoGenerator:
    """
    Comprehensive video generator for viral content
    Features:
    - Multiple video styles
    - Text-to-speech narration
    - Background video/image support
    - Animated text and effects
    - Platform-specific optimizations
    """
    
    def __init__(self, assets_dir: str = 'assets'):
        self.assets_dir = assets_dir
        self.logger = logging.getLogger('video_generator')
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self._setup_tts()
        
        # Font configurations
        self.fonts = {
            'title': os.path.join(assets_dir, 'fonts', 'Montserrat-Bold.ttf'),
            'body': os.path.join(assets_dir, 'fonts', 'OpenSans-Regular.ttf'),
            'caption': os.path.join(assets_dir, 'fonts', 'Roboto-Medium.ttf')
        }
        
        # Color schemes
        self.color_schemes = {
            'modern': {
                'primary': '#FFFFFF',
                'secondary': '#FFD700',
                'background': '#000000',
                'accent': '#FF4444'
            },
            'minimal': {
                'primary': '#000000',
                'secondary': '#666666',
                'background': '#FFFFFF',
                'accent': '#0066CC'
            },
            'gaming': {
                'primary': '#00FF00',
                'secondary': '#FF00FF',
                'background': '#0A0A0A',
                'accent': '#00FFFF'
            },
            'meme': {
                'primary': '#FFFFFF',
                'secondary': '#000000',
                'background': '#FF0000',
                'accent': '#FFFF00'
            }
        }
        
        # Background video categories
        self.background_categories = {
            'gaming': ['minecraft_parkour', 'subway_surfers', 'gta_driving', 'fortnite_gameplay'],
            'nature': ['ocean_waves', 'forest_walk', 'sunset_timelapse', 'rain_window'],
            'abstract': ['particles', 'gradient_flow', 'geometric_shapes', 'liquid_motion'],
            'lifestyle': ['coffee_pour', 'city_lights', 'cooking_prep', 'workout_gym']
        }
    
    def _setup_tts(self):
        """Setup text-to-speech engine"""
        # Set properties
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        
        # Try to set a good voice
        for voice in voices:
            if 'english' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
    
    def create_video(self, script: Dict, config: Optional[VideoConfig] = None) -> str:
        """
        Create a video from script
        
        Args:
            script: Dictionary containing:
                - segments: List of video segments
                - title: Video title
                - style: Visual style
                - music: Background music info
            config: Video configuration
        
        Returns:
            Path to generated video file
        """
        if not config:
            config = VideoConfig()
        
        try:
            # Generate audio narration
            audio_path = self._generate_narration(script)
            
            # Get background video
            background = self._get_background_video(script, config)
            
            # Create text overlays
            text_clips = self._create_text_overlays(script, config)
            
            # Add visual effects
            effects_clips = self._create_visual_effects(script, config)
            
            # Combine all elements
            final_video = self._compose_video(
                background, 
                audio_path, 
                text_clips, 
                effects_clips,
                config
            )
            
            # Add music
            if script.get('music'):
                final_video = self._add_background_music(final_video, script['music'])
            
            # Export video
            output_path = self._export_video(final_video, script, config)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            raise
    
    def _generate_narration(self, script: Dict) -> str:
        """Generate audio narration from script"""
        segments = script.get('segments', [])
        
        # Combine all narration text
        narration_text = ' '.join([
            seg['narration'] 
            for seg in segments 
            if seg.get('narration')
        ])
        
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        
        # Use gTTS for better quality
        try:
            tts = gTTS(text=narration_text, lang='en', slow=False)
            tts.save(temp_audio.name)
        except:
            # Fallback to pyttsx3
            self.tts_engine.save_to_file(narration_text, temp_audio.name)
            self.tts_engine.runAndWait()
        
        return temp_audio.name
    
    def _get_background_video(self, script: Dict, config: VideoConfig) -> VideoClip:
        """Get appropriate background video"""
        style = script.get('style', 'modern')
        
        # Determine background category
        if style == 'gaming' or 'minecraft' in script.get('title', '').lower():
            category = 'gaming'
            video_name = 'minecraft_parkour'
        elif style == 'minimal':
            category = 'abstract'
            video_name = random.choice(self.background_categories[category])
        else:
            category = 'lifestyle'
            video_name = random.choice(self.background_categories[category])
        
        # Load background video
        background_path = os.path.join(
            self.assets_dir, 
            'backgrounds', 
            category, 
            f'{video_name}.mp4'
        )
        
        if os.path.exists(background_path):
            # Load and loop video
            background = VideoFileClip(background_path)
            
            # Resize to fit config
            background = background.resize((config.width, config.height))
            
            # Loop if necessary
            if background.duration < config.duration:
                background = background.loop(duration=config.duration)
            else:
                background = background.subclip(0, config.duration)
            
            # Apply style-specific filters
            if style == 'modern':
                background = background.fx(vfx.colorx, 0.8)  # Darken
            elif style == 'minimal':
                background = background.fx(vfx.blackwhite)  # Black and white
            
            return background
        else:
            # Create gradient background as fallback
            return self._create_gradient_background(config)
    
    def _create_gradient_background(self, config: VideoConfig) -> VideoClip:
        """Create animated gradient background"""
        def make_frame(t):
            # Create gradient that changes over time
            gradient = np.zeros((config.height, config.width, 3))
            
            # Animated gradient
            for y in range(config.height):
                progress = y / config.height
                r = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 10)))
                g = int(255 * progress)
                b = int(255 * (1 - progress))
                gradient[y, :] = [r, g, b]
            
            return gradient.astype('uint8')
        
        return VideoClip(make_frame, duration=config.duration)
    
    def _create_text_overlays(self, script: Dict, config: VideoConfig) -> List[VideoClip]:
        """Create animated text overlays"""
        text_clips = []
        colors = self.color_schemes.get(script.get('style', 'modern'))
        
        # Title card
        if script.get('title'):
            title_clip = self._create_title_card(
                script['title'],
                config,
                colors
            )
            text_clips.append(title_clip)
        
        # Process each segment
        current_time = 3  # Start after title
        
        for i, segment in enumerate(script.get('segments', [])):
            # Main text
            if segment.get('text'):
                text_clip = self._create_animated_text(
                    segment['text'],
                    start_time=current_time,
                    duration=segment.get('duration', 5),
                    position='center',
                    style=script.get('style', 'modern'),
                    config=config,
                    colors=colors
                )
                text_clips.append(text_clip)
            
            # Captions
            if segment.get('caption'):
                caption_clip = self._create_caption(
                    segment['caption'],
                    start_time=current_time,
                    duration=segment.get('duration', 5),
                    config=config,
                    colors=colors
                )
                text_clips.append(caption_clip)
            
            current_time += segment.get('duration', 5)
        
        # End card
        if script.get('call_to_action'):
            end_card = self._create_end_card(
                script['call_to_action'],
                start_time=current_time,
                config=config,
                colors=colors
            )
            text_clips.append(end_card)
        
        return text_clips
    
    def _create_title_card(self, title: str, config: VideoConfig, colors: Dict) -> VideoClip:
        """Create animated title card"""
        # Create title text
        title_clip = TextClip(
            title,
            fontsize=80,
            color=colors['primary'],
            font=self.fonts.get('title', 'Arial'),
            method='caption',
            size=(config.width * 0.8, None),
            align='center'
        )
        
        # Add shadow
        shadow = TextClip(
            title,
            fontsize=80,
            color='black',
            font=self.fonts.get('title', 'Arial'),
            method='caption',
            size=(config.width * 0.8, None),
            align='center'
        ).set_position((5, 5), relative=True)
        
        # Combine with animation
        title_composite = CompositeVideoClip([
            shadow.set_opacity(0.5),
            title_clip
        ])
        
        # Animate: fade in, stay, fade out
        title_animated = (title_composite
            .set_position('center')
            .set_duration(3)
            .crossfadein(0.5)
            .crossfadeout(0.5))
        
        return title_animated
    
    def _create_animated_text(self, text: str, start_time: float, duration: float,
                            position: str, style: str, config: VideoConfig, 
                            colors: Dict) -> VideoClip:
        """Create animated text with style-specific effects"""
        
        # Base text clip
        text_clip = TextClip(
            text,
            fontsize=60,
            color=colors['primary'],
            font=self.fonts.get('body', 'Arial'),
            method='caption',
            size=(config.width * 0.9, None),
            align='center'
        )
        
        # Apply style-specific animations
        if style == 'modern':
            # Slide in from bottom
            animated = text_clip.set_position(
                lambda t: ('center', config.height - 100 - t * 200) if t < 0.5 else ('center', config.height - 200)
            )
        elif style == 'minimal':
            # Fade in
            animated = text_clip.crossfadein(0.5)
        elif style == 'gaming':
            # Glitch effect
            animated = self._add_glitch_effect(text_clip)
        elif style == 'meme':
            # Impact font style with stroke
            animated = self._create_meme_text(text, config, colors)
        else:
            animated = text_clip
        
        # Set timing
        animated = animated.set_start(start_time).set_duration(duration)
        
        return animated
    
    def _create_caption(self, caption: str, start_time: float, duration: float,
                       config: VideoConfig, colors: Dict) -> VideoClip:
        """Create caption/subtitle"""
        caption_clip = TextClip(
            caption,
            fontsize=40,
            color=colors['secondary'],
            font=self.fonts.get('caption', 'Arial'),
            method='caption',
            size=(config.width * 0.8, None),
            align='center',
            bg_color='black'
        )
        
        # Position at bottom
        positioned = caption_clip.set_position(('center', config.height - 150))
        
        # Set timing
        return positioned.set_start(start_time).set_duration(duration)
    
    def _create_visual_effects(self, script: Dict, config: VideoConfig) -> List[VideoClip]:
        """Create visual effects based on script"""
        effects = []
        style = script.get('style', 'modern')
        
        # Style-specific effects
        if style == 'gaming':
            # Add gaming overlays
            effects.extend(self._create_gaming_effects(config))
        elif style == 'modern':
            # Add modern transitions
            effects.extend(self._create_modern_effects(config))
        elif style == 'meme':
            # Add meme elements
            effects.extend(self._create_meme_effects(script, config))
        
        # Add progress bar
        if script.get('show_progress', True):
            progress_bar = self._create_progress_bar(config)
            effects.append(progress_bar)
        
        return effects
    
    def _create_gaming_effects(self, config: VideoConfig) -> List[VideoClip]:
        """Create gaming-style visual effects"""
        effects = []
        
        # Health bar
        health_bar = self._create_animated_bar(
            position=(50, 50),
            size=(200, 30),
            color='red',
            duration=config.duration
        )
        effects.append(health_bar)
        
        # Score counter
        score_counter = self._create_score_counter(
            position=(config.width - 200, 50),
            duration=config.duration
        )
        effects.append(score_counter)
        
        return effects
    
    def _create_progress_bar(self, config: VideoConfig) -> VideoClip:
        """Create video progress bar"""
        def make_frame(t):
            # Create progress bar image
            img = Image.new('RGBA', (config.width, 10), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Background
            draw.rectangle([0, 0, config.width, 10], fill=(255, 255, 255, 50))
            
            # Progress
            progress = t / config.duration
            draw.rectangle([0, 0, int(config.width * progress), 10], fill=(255, 215, 0, 255))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=config.duration).set_position(('center', 'bottom'))
    
    def _compose_video(self, background: VideoClip, audio_path: str,
                      text_clips: List[VideoClip], effects: List[VideoClip],
                      config: VideoConfig) -> VideoClip:
        """Compose all elements into final video"""
        
        # Load audio
        audio = AudioFileClip(audio_path)
        
        # Adjust video duration to match audio
        if audio.duration != config.duration:
            config.duration = audio.duration
            background = background.subclip(0, config.duration)
        
        # Combine all clips
        all_clips = [background] + text_clips + effects
        
        # Create composite
        final_video = CompositeVideoClip(all_clips, size=(config.width, config.height))
        
        # Add audio
        final_video = final_video.set_audio(audio)
        
        return final_video
    
    def _add_background_music(self, video: VideoClip, music_info: Dict) -> VideoClip:
        """Add background music to video"""
        music_path = os.path.join(
            self.assets_dir,
            'music',
            music_info.get('filename', 'default.mp3')
        )
        
        if os.path.exists(music_path):
            # Load music
            music = AudioFileClip(music_path)
            
            # Adjust volume
            music = music.volumex(music_info.get('volume', 0.3))
            
            # Loop if necessary
            if music.duration < video.duration:
                music = music.loop(duration=video.duration)
            else:
                music = music.subclip(0, video.duration)
            
            # Mix with existing audio
            composite_audio = CompositeAudioClip([video.audio, music])
            
            return video.set_audio(composite_audio)
        
        return video
    
    def _export_video(self, video: VideoClip, script: Dict, config: VideoConfig) -> str:
        """Export video with platform-specific settings"""
        
        # Create output filename
        timestamp = int(time.time())
        title_slug = script.get('title', 'video').lower().replace(' ', '_')[:30]
        filename = f"{title_slug}_{config.platform}_{timestamp}.mp4"
        output_path = os.path.join('output', 'videos', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Platform-specific export settings
        if config.platform == 'tiktok':
            codec = 'libx264'
            bitrate = '8000k'
            audio_codec = 'aac'
            audio_bitrate = '128k'
        elif config.platform == 'youtube':
            codec = 'libx264'
            bitrate = '10000k'
            audio_codec = 'aac'
            audio_bitrate = '192k'
        elif config.platform == 'instagram':
            codec = 'libx264'
            bitrate = '8000k'
            audio_codec = 'aac'
            audio_bitrate = '128k'
        else:
            codec = 'libx264'
            bitrate = '8000k'
            audio_codec = 'aac'
            audio_bitrate = '128k'
        
        # Export with optimal settings
        video.write_videofile(
            output_path,
            codec=codec,
            bitrate=bitrate,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            fps=config.fps,
            preset='medium',
            threads=4
        )
        
        self.logger.info(f"Video exported to: {output_path}")
        
        return output_path
    
    def _add_glitch_effect(self, clip: VideoClip) -> VideoClip:
        """Add glitch effect to clip"""
        def glitch_frame(get_frame, t):
            frame = get_frame(t)
            
            # Random glitch probability
            if random.random() < 0.1:  # 10% chance
                # Shift RGB channels
                h, w = frame.shape[:2]
                shift = random.randint(5, 20)
                
                # Create glitched frame
                glitched = frame.copy()
                glitched[:, shift:, 0] = frame[:, :-shift, 0]  # Shift red
                glitched[:, :-shift, 1] = frame[:, shift:, 1]  # Shift green
                
                return glitched
            
            return frame
        
        return clip.fl(glitch_frame)
    
    def _create_meme_text(self, text: str, config: VideoConfig, colors: Dict) -> VideoClip:
        """Create meme-style text with stroke"""
        # White text with black stroke (Impact font style)
        
        # Create text with stroke effect by layering
        stroke_width = 3
        clips = []
        
        # Create stroke by offsetting black text
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:
                    stroke_clip = TextClip(
                        text.upper(),
                        fontsize=70,
                        color='black',
                        font='Impact',
                        method='caption',
                        size=(config.width * 0.9, None),
                        align='center'
                    ).set_position((dx, dy), relative=True)
                    clips.append(stroke_clip)
        
        # Main text on top
        main_text = TextClip(
            text.upper(),
            fontsize=70,
            color='white',
            font='Impact',
            method='caption',
            size=(config.width * 0.9, None),
            align='center'
        )
        clips.append(main_text)
        
        # Composite all layers
        return CompositeVideoClip(clips)
    
    def _create_animated_bar(self, position: Tuple[int, int], size: Tuple[int, int],
                           color: str, duration: float) -> VideoClip:
        """Create animated bar (health, mana, etc.)"""
        def make_frame(t):
            img = Image.new('RGBA', size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Background
            draw.rectangle([0, 0, size[0], size[1]], fill=(50, 50, 50, 200))
            
            # Animated fill
            fill_width = int(size[0] * (0.3 + 0.7 * abs(np.sin(t))))
            draw.rectangle([0, 0, fill_width, size[1]], fill=color)
            
            # Border
            draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(255, 255, 255, 255), width=2)
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration).set_position(position)
    
    def _create_score_counter(self, position: Tuple[int, int], duration: float) -> VideoClip:
        """Create animated score counter"""
        def make_frame(t):
            # Calculate score based on time
            score = int(t * 1000)
            
            # Create text
            text_clip = TextClip(
                f"SCORE: {score:,}",
                fontsize=40,
                color='white',
                font='Arial-Bold'
            )
            
            # Get frame
            frame = text_clip.get_frame(0)
            
            return frame
        
        clip = VideoClip(make_frame, duration=duration)
        return clip.set_position(position)
    
    def _create_modern_effects(self, config: VideoConfig) -> List[VideoClip]:
        """Create modern visual effects"""
        effects = []
        
        # Animated circles
        for i in range(3):
            circle = self._create_animated_circle(
                position=(random.randint(100, config.width-100), 
                         random.randint(100, config.height-100)),
                radius=random.randint(50, 150),
                color=(255, 215, 0, 50),  # Gold with transparency
                duration=config.duration
            )
            effects.append(circle)
        
        # Particles
        particles = self._create_particle_effect(config)
        effects.append(particles)
        
        return effects
    
    def _create_animated_circle(self, position: Tuple[int, int], radius: int,
                              color: Tuple[int, int, int, int], duration: float) -> VideoClip:
        """Create animated circle effect"""
        def make_frame(t):
            img = Image.new('RGBA', (radius*2, radius*2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Pulsing effect
            current_radius = int(radius * (0.8 + 0.2 * np.sin(2 * np.pi * t / 3)))
            
            # Draw circle
            draw.ellipse(
                [radius-current_radius, radius-current_radius,
                 radius+current_radius, radius+current_radius],
                fill=color
            )
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration).set_position(position)
    
    def _create_particle_effect(self, config: VideoConfig) -> VideoClip:
        """Create particle effect overlay"""
        particles = []
        num_particles = 20
        
        # Initialize particles
        for _ in range(num_particles):
            particles.append({
                'x': random.randint(0, config.width),
                'y': random.randint(0, config.height),
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-5, -1),
                'size': random.randint(2, 6),
                'lifetime': random.uniform(0, config.duration)
            })
        
        def make_frame(t):
            img = Image.new('RGBA', (config.width, config.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            for particle in particles:
                # Update position
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                
                # Reset if out of bounds
                if particle['y'] < 0:
                    particle['y'] = config.height
                    particle['x'] = random.randint(0, config.width)
                
                # Draw particle
                opacity = int(255 * (1 - abs(t - particle['lifetime']) / config.duration))
                draw.ellipse(
                    [particle['x']-particle['size'], particle['y']-particle['size'],
                     particle['x']+particle['size'], particle['y']+particle['size']],
                    fill=(255, 255, 255, opacity)
                )
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=config.duration)
    
    def _create_meme_effects(self, script: Dict, config: VideoConfig) -> List[VideoClip]:
        """Create meme-specific effects"""
        effects = []
        
        # Add emoji reactions
        emojis = ['😂', '💀', '🔥', '💯', '😭']
        for i, emoji in enumerate(random.sample(emojis, 3)):
            emoji_clip = self._create_floating_emoji(
                emoji,
                start_time=i * config.duration / 3,
                config=config
            )
            effects.append(emoji_clip)
        
        # Add MLG airhorn moments
        if 'epic' in script.get('title', '').lower():
            airhorn_effect = self._create_mlg_effect(config)
            effects.append(airhorn_effect)
        
        return effects
    
    def _create_floating_emoji(self, emoji: str, start_time: float, 
                              config: VideoConfig) -> VideoClip:
        """Create floating emoji effect"""
        emoji_clip = TextClip(
            emoji,
            fontsize=80,
            font='Segoe UI Emoji'
        )
        
        # Animate position
        def position_func(t):
            x = config.width * 0.8
            y = config.height - (t * 200)  # Float upward
            return (x, y)
        
        animated = emoji_clip.set_position(position_func)
        
        return animated.set_start(start_time).set_duration(3).crossfadeout(0.5)
    
    def _create_mlg_effect(self, config: VideoConfig) -> VideoClip:
        """Create MLG-style effect"""
        # Create flashing overlay
        def make_frame(t):
            img = Image.new('RGBA', (config.width, config.height), (0, 0, 0, 0))
            
            # Flash effect at specific moments
            if int(t * 10) % 10 == 0:
                draw = ImageDraw.Draw(img)
                # Rainbow colors
                colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), 
                         (0, 255, 0), (0, 0, 255), (75, 0, 130)]
                
                section_height = config.height // len(colors)
                for i, color in enumerate(colors):
                    draw.rectangle(
                        [0, i * section_height, config.width, (i + 1) * section_height],
                        fill=(*color, 50)
                    )
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=config.duration)
    
    def _create_end_card(self, cta: str, start_time: float, 
                        config: VideoConfig, colors: Dict) -> VideoClip:
        """Create end card with call-to-action"""
        # Background
        bg_clip = ColorClip(
            size=(config.width, config.height),
            color=(0, 0, 0),
            duration=5
        ).set_opacity(0.8)
        
        # CTA text
        cta_text = TextClip(
            cta,
            fontsize=60,
            color=colors['primary'],
            font=self.fonts.get('title', 'Arial'),
            method='caption',
            size=(config.width * 0.8, None),
            align='center'
        ).set_position('center')
        
        # Follow button
        follow_button = self._create_follow_button(config, colors)
        
        # Combine
        end_card = CompositeVideoClip([
            bg_clip,
            cta_text,
            follow_button
        ])
        
        return end_card.set_start(start_time).set_duration(5)
    
    def _create_follow_button(self, config: VideoConfig, colors: Dict) -> VideoClip:
        """Create animated follow button"""
        def make_frame(t):
            img = Image.new('RGBA', (300, 80), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Pulsing effect
            scale = 1 + 0.1 * np.sin(2 * np.pi * t)
            
            # Button background
            draw.rounded_rectangle(
                [10, 10, 290, 70],
                radius=40,
                fill=colors['accent']
            )
            
            return np.array(img)
        
        button_bg = VideoClip(make_frame, duration=5)
        
        # Button text
        button_text = TextClip(
            "FOLLOW",
            fontsize=40,
            color='white',
            font=self.fonts.get('title', 'Arial-Bold')
        ).set_position('center')
        
        # Combine
        button = CompositeVideoClip([button_bg, button_text], size=(300, 80))
        
        return button.set_position(('center', config.height - 200))


# Example usage
if __name__ == '__main__':
    # Example script
    script = {
        'title': 'Amazing Reddit Story',
        'style': 'modern',
        'segments': [
            {
                'text': 'This incredible story will blow your mind!',
                'narration': 'Let me tell you about something amazing that happened on Reddit.',
                'duration': 5
            },
            {
                'text': 'A user posted about finding $10,000 in their attic',
                'narration': 'One day, a Reddit user was cleaning their attic when they made an incredible discovery.',
                'caption': 'True story from r/AskReddit',
                'duration': 6
            },
            {
                'text': 'But what happened next was even more surprising...',
                'narration': 'But the story doesn\'t end there. What they found next changed everything.',
                'duration': 4
            }
        ],
        'call_to_action': 'Follow for more amazing stories!',
        'music': {
            'filename': 'upbeat_background.mp3',
            'volume': 0.3
        }
    }
    
    # Create video
    generator = VideoGenerator()
    config = VideoConfig(
        platform='tiktok',
        style='modern',
        duration=15
    )
    
    video_path = generator.create_video(script, config)
    print(f"Video created: {video_path}")