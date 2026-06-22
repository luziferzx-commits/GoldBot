import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EconomicCalendar:
    """
    Fetches and manages economic calendar events.
    """
    
    def __init__(self):
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.events_df = pd.DataFrame()
        self.api_error = False
        self.last_fetch = None

    def fetch_news(self) -> bool:
        """
        Fetch the news calendar from the API.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            logger.info(f"Fetching economic calendar from {self.url}...")
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.api_error = True
                return False
                
            self.api_error = False
                
            self.events_df = pd.DataFrame(data)
            # Parse datetime: Format usually looks like "2025-01-20T08:30:00-05:00"
            self.events_df['date'] = pd.to_datetime(self.events_df['date'], utc=True)
            self.last_fetch = datetime.utcnow()
            logger.info(f"Loaded {len(self.events_df)} news events.")
            return True
        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")
            self.api_error = True
            self.last_fetch = datetime.utcnow() # Note the error time
            return False

    def is_news_time(self, minutes_before: int = 30, minutes_after: int = 15) -> bool:
        """
        Check if current time is within a restricted window around High Impact news.
        
        Args:
            minutes_before: Window before news in minutes.
            minutes_after: Window after news in minutes.
            
        Returns:
            bool: True if inside news restricted window, False otherwise.
        """
        # Auto-refetch if older than 12 hours
        if self.last_fetch is None or datetime.utcnow() - self.last_fetch > timedelta(hours=12):
            self.fetch_news()
            
        if self.api_error:
            logger.warning("News API Error: Fallback mode active (Trading disabled)")
            return True
            
        if self.events_df.empty:
            return False
            
        now = datetime.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
        
        # Filter for High Impact news only (and specific currency like USD if desired)
        high_impact = self.events_df[self.events_df['impact'] == 'High']
        
        for _, row in high_impact.iterrows():
            news_time = row['date']
            window_start = news_time - timedelta(minutes=minutes_before)
            window_end = news_time + timedelta(minutes=minutes_after)
            
            if window_start <= now <= window_end:
                logger.warning(f"Restricted news time! Event: {row['title']} at {news_time}")
                return True
                
        return False

    def get_upcoming_news(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get a list of upcoming news events within the specified hours.
        
        Args:
            hours: Look-ahead window in hours.
            
        Returns:
            List of news dictionaries.
        """
        if self.events_df.empty:
            return []
            
        now = datetime.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
        horizon = now + timedelta(hours=hours)
        
        upcoming = self.events_df[(self.events_df['date'] >= now) & (self.events_df['date'] <= horizon)]
        return upcoming.to_dict('records')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    calendar = EconomicCalendar()
    if calendar.fetch_news():
        print(f"Is news time now? {calendar.is_news_time()}")
        upcoming = calendar.get_upcoming_news(hours=24)
        print(f"Upcoming events in 24h: {len(upcoming)}")
        if upcoming:
            for ev in upcoming[:3]:
                print(f"- {ev['date']}: [{ev['impact']}] {ev['country']} {ev['title']}")
