# auth/session_manager.py
"""
Session management for user authentication
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis
import json
import logging
from werkzeug.security import generate_password_hash, check_password_hash

from config import get_config
from database.connection import get_session
from database.models import User

class SessionManager:
    """Manages user sessions and authentication"""
    
    def __init__(self, redis_client=None):
        self.config = get_config().security
        self.redis = redis_client or redis.Redis(
            host=get_config().redis.host,
            port=get_config().redis.port,
            password=get_config().redis.password,
            db=get_config().redis.db + 1,  # Use different DB for sessions
            decode_responses=True
        )
        self.logger = logging.getLogger('session_manager')
        
        # Session settings
        self.session_timeout = timedelta(hours=self.config.jwt_expiry_hours)
        self.session_prefix = "session:"
        
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str = None, role: str = 'user') -> bool:
        """Create a new user account"""
        try:
            with get_session() as session:
                # Check if user exists
                existing = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing:
                    return False
                
                # Create new user
                user = User(
                    username=username,
                    email=email,
                    password_hash=generate_password_hash(password),
                    full_name=full_name,
                    role=role
                )
                
                session.add(user)
                session.commit()
                
                self.logger.info(f"Created user: {username}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and create session"""
        try:
            with get_session() as session:
                user = session.query(User).filter(
                    (User.username == username) | (User.email == username)
                ).first()
                
                if not user or not user.active:
                    return None
                
                if not check_password_hash(user.password_hash, password):
                    return None
                
                # Update last login
                user.last_login = datetime.utcnow()
                session.commit()
                
                # Create session
                session_id = self._generate_session_id()
                session_data = {
                    'user_id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'created_at': time.time(),
                    'last_activity': time.time()
                }
                
                # Store in Redis
                session_key = f"{self.session_prefix}{session_id}"
                self.redis.setex(
                    session_key,
                    int(self.session_timeout.total_seconds()),
                    json.dumps(session_data)
                )
                
                self.logger.info(f"User {username} logged in")
                
                return {
                    'session_id': session_id,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'role': user.role,
                        'full_name': user.full_name
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        try:
            session_key = f"{self.session_prefix}{session_id}"
            session_data = self.redis.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            
            # Update last activity
            data['last_activity'] = time.time()
            self.redis.setex(
                session_key,
                int(self.session_timeout.total_seconds()),
                json.dumps(data)
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get session: {e}")
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        try:
            session_key = f"{self.session_prefix}{session_id}"
            self.redis.delete(session_key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate session: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions (Redis handles this automatically)"""
        # Redis TTL handles expiration, but we can log active sessions
        try:
            pattern = f"{self.session_prefix}*"
            active_sessions = len(self.redis.keys(pattern))
            self.logger.debug(f"Active sessions: {active_sessions}")
        except Exception as e:
            self.logger.error(f"Session cleanup error: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)

# auth/token_manager.py
"""
JWT token management for API authentication
"""

import jwt
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from config import get_config

class TokenManager:
    """Manages JWT tokens for API authentication"""
    
    def __init__(self):
        self.config = get_config().security
        self.logger = logging.getLogger('token_manager')
        self.algorithm = 'HS256'
        
    def generate_token(self, user_data: Dict, expires_in: Optional[int] = None) -> str:
        """Generate a JWT token"""
        expires_in = expires_in or (self.config.jwt_expiry_hours * 3600)
        
        payload = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'role': user_data.get('role'),
            'iat': int(time.time()),
            'exp': int(time.time() + expires_in)
        }
        
        token = jwt.encode(payload, self.config.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh a JWT token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        # Remove timing fields
        payload.pop('iat', None)
        payload.pop('exp', None)
        
        # Generate new token
        return self.generate_token(payload)

# security/encryption.py
"""
Data encryption utilities
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

class EncryptionManager:
    """Handles data encryption and decryption"""
    
    def __init__(self, password: str = None):
        self.logger = logging.getLogger('encryption')
        self.password = password or os.getenv('ENCRYPTION_PASSWORD', 'default-password')
        self._fernet = None
    
    @property
    def fernet(self):
        """Get or create Fernet instance"""
        if self._fernet is None:
            # Derive key from password
            password_bytes = self.password.encode()
            salt = b'scrapehive_salt'  # In production, use random salt
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self._fernet = Fernet(key)
        
        return self._fernet
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: dict) -> str:
        """Encrypt dictionary data"""
        import json
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> dict:
        """Decrypt dictionary data"""
        import json
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)

# security/input_validation.py
"""
Input validation and sanitization
"""

import re
import html
from typing import Any, Dict, List
from urllib.parse import urlparse
import bleach
import logging

class InputValidator:
    """Validates and sanitizes user input"""
    
    def __init__(self):
        self.logger = logging.getLogger('input_validator')
        
        # Allowed HTML tags for rich text
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]
        
        # Common patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'username': re.compile(r'^[a-zA-Z0-9_]{3,30}$'),
            'node_id': re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
            'session_id': re.compile(r'^[a-zA-Z0-9_-]{32,}$'),
        }
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        return bool(self.patterns['email'].match(email))
    
    def validate_username(self, username: str) -> bool:
        """Validate username format"""
        return bool(self.patterns['username'].match(username))
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def sanitize_string(self, text: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(text, str):
            return str(text)
        
        # Limit length
        text = text[:max_length]
        
        # HTML escape
        text = html.escape(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content"""
        return bleach.clean(
            html_content,
            tags=self.allowed_tags,
            strip=True
        )
    
    def validate_dict(self, data: Dict, schema: Dict) -> Dict:
        """Validate dictionary against schema"""
        validated = {}
        
        for key, rules in schema.items():
            if key not in data and rules.get('required', False):
                raise ValueError(f"Missing required field: {key}")
            
            if key not in data:
                continue
            
            value = data[key]
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}")
            
            # String validation
            if isinstance(value, str):
                max_length = rules.get('max_length', 1000)
                value = self.sanitize_string(value, max_length)
                
                pattern = rules.get('pattern')
                if pattern and not pattern.match(value):
                    raise ValueError(f"Invalid format for {key}")
            
            # Numeric validation
            elif isinstance(value, (int, float)):
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None and value < min_val:
                    raise ValueError(f"{key} must be >= {min_val}")
                if max_val is not None and value > max_val:
                    raise ValueError(f"{key} must be <= {max_val}")
            
            validated[key] = value
        
        return validated

# Example schemas for common data
VALIDATION_SCHEMAS = {
    'user_registration': {
        'username': {
            'type': str,
            'required': True,
            'pattern': re.compile(r'^[a-zA-Z0-9_]{3,30}$'),
            'max_length': 30
        },
        'email': {
            'type': str,
            'required': True,
            'pattern': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'max_length': 100
        },
        'password': {
            'type': str,
            'required': True,
            'max_length': 128
        },
        'full_name': {
            'type': str,
            'required': False,
            'max_length': 100
        }
    },
    'task_creation': {
        'type': {
            'type': str,
            'required': True,
            'max_length': 50
        },
        'target': {
            'type': str,
            'required': True,
            'max_length': 2000
        },
        'priority': {
            'type': int,
            'required': False,
            'min': 1,
            'max': 10
        }
    }
}