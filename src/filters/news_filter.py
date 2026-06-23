import logging
import yaml
import requests
import json
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class NewsFilter:
    """
    Fetches economic calendar data and blocks trading around high-impact news.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.enabled = False
        self._cached_events = []
        self._cache_time = None
        
        try:
            with open(config_path, "r") as f:
                settings = yaml.safe_load(f)
            
            news_config = settings.get('news_filter', {})
            self.enabled = news_config.get('enabled', False)
            self.block_before = timedelta(minutes=news_config.get('block_minutes_before', 30))
            self.block_after = timedelta(minutes=news_config.get('block_minutes_after', 15))
            self.impact_levels = [i.upper() for i in news_config.get('impact_levels', ['HIGH'])]
            self.currencies = [c.upper() for c in news_config.get('currencies', ['USD', 'XAU'])]
            self.url = news_config.get('calendar_url', "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
            self.cache_ttl = timedelta(minutes=news_config.get('cache_ttl_minutes', 60))
            
        except Exception as e:
            logger.error(f"Failed to load news filter config: {e}")

    def is_news_window(self, check_time: datetime = None) -> Tuple[bool, str]:
        """
        Check if the given time (UTC) is within the block window of any high impact news.
        """
        if not self.enabled:
            return False, ""
            
        now = check_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
            
        events = self._get_events()
        
        if not events:
            # Fail open if we cannot fetch news
            return False, ""
            
        for event in events:
            event_time = event.get('_dt')
            if not event_time:
                continue
                
            window_start = event_time - self.block_before
            window_end = event_time + self.block_after
            
            if window_start <= now <= window_end:
                title = event.get('title', 'Unknown News')
                time_str = event_time.strftime('%Y-%m-%d %H:%M UTC')
                return True, f"News window: {title} at {time_str}"
                
        return False, ""

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[dict]:
        """Return events occurring within the next N hours."""
        if not self.enabled:
            return []
            
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=hours_ahead)
        
        events = self._get_events()
        upcoming = []
        
        for e in events:
            dt = e.get('_dt')
            if dt and now <= dt <= end_time:
                upcoming.append(e)
                
        return upcoming

    def _get_events(self) -> List[dict]:
        now = datetime.now(timezone.utc)
        if self._cache_time is None or (now - self._cache_time) > self.cache_ttl:
            self._cached_events = self._fetch_calendar()
            self._cache_time = now
        return self._cached_events

    def _fetch_calendar(self) -> List[dict]:
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            filtered = []
            for item in data:
                country = str(item.get('country', '')).upper()
                impact = str(item.get('impact', '')).upper()
                
                if country in self.currencies and impact in self.impact_levels:
                    dt = self._parse_event_time(item)
                    if dt:
                        item['_dt'] = dt
                        filtered.append(item)
            
            logger.debug(f"Fetched {len(filtered)} relevant news events.")
            return filtered
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []

    def _parse_event_time(self, event: dict) -> Optional[datetime]:
        date_str = event.get('date', '')
        time_str = event.get('time', '')
        
        if not date_str or not time_str or time_str.lower() in ['all day', 'day 1', 'day 2', 'day 3']:
            return None
            
        try:
            # Parse '2023-11-30' and '8:30am'
            dt_str = f"{date_str} {time_str}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %I:%M%p")
            
            # ForexFactory typically provides US Eastern time for this endpoint without tz.
            # Convert US Eastern (approx UTC-5) to UTC (+5 hours). 
            # Note: For production, pytz/zoneinfo with 'US/Eastern' should handle Daylight Savings properly.
            dt_utc = dt + timedelta(hours=5)
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            
            return dt_utc
        except Exception as e:
            logger.debug(f"Could not parse event time '{date_str} {time_str}': {e}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    nf = NewsFilter()
    blocked, reason = nf.is_news_window()
    print(f"Blocked: {blocked}, Reason: {reason}")
    print("Upcoming events:")
    for e in nf.get_upcoming_events():
        print(e['title'], e['_dt'])
