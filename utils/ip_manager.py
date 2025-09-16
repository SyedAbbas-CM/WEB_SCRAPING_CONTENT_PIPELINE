"""
IP Manager stub for selecting IP/proxy per platform. Returns no-proxy by default.
"""

from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class IPConfig:
    address: str
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = 'http'
    
    def to_proxy_dict(self) -> Dict[str, str]:
        auth = ''
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        url = f"{self.protocol}://{auth}{self.address}:{self.port}"
        return {'http': url, 'https': url}

class IPType:
    RESIDENTIAL = 'residential'
    DATACENTER = 'datacenter'
    MOBILE = 'mobile'

class IPManager:
    def get_ip_for_platform(self, platform: str) -> Optional[IPConfig]:
        # In a real implementation, select an appropriate IP/proxy
        return None
    
    def mark_ip_blocked(self, ip_config: IPConfig, platform: str):
        # Record block event and deprioritize this IP
        pass

