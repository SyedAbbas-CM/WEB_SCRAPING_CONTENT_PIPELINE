"""
SessionManager: persistent cookie sessions per domain with simple helpers.
"""

import os
import time
from http.cookiejar import LWPCookieJar
from typing import Optional, Dict
import requests


class SessionManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.expanduser('~/.scrapehive/cookies')
        os.makedirs(self.base_dir, exist_ok=True)
        self._sessions: Dict[str, requests.Session] = {}

    def _cookie_path(self, domain: str) -> str:
        safe = domain.replace(':', '_').replace('/', '_')
        return os.path.join(self.base_dir, f'{safe}.lwp')

    def get_session(self, domain: str, proxy: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Session:
        if domain in self._sessions:
            return self._sessions[domain]
        s = requests.Session()
        # Attach cookies
        cookie_path = self._cookie_path(domain)
        s.cookies = LWPCookieJar(cookie_path)
        try:
            s.cookies.load(ignore_discard=True)
        except Exception:
            pass
        # Proxies
        if proxy and proxy.get('url'):
            s.proxies = {'http': proxy['url'], 'https': proxy['url']}
        # Headers
        if headers:
            s.headers.update(headers)
        self._sessions[domain] = s
        return s

    def save(self, domain: Optional[str] = None):
        if domain:
            s = self._sessions.get(domain)
            if s and isinstance(s.cookies, LWPCookieJar):
                try:
                    s.cookies.save(ignore_discard=True)
                except Exception:
                    pass
            return
        for s in self._sessions.values():
            if isinstance(s.cookies, LWPCookieJar):
                try:
                    s.cookies.save(ignore_discard=True)
                except Exception:
                    pass

    def import_cookies(self, domain: str, cookie_file: str):
        s = self.get_session(domain)
        try:
            jar = LWPCookieJar(cookie_file)
            jar.load(ignore_discard=True)
            for c in jar:
                s.cookies.set_cookie(c)
            self.save(domain)
        except Exception:
            pass

