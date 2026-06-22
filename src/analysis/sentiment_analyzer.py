import feedparser
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes live news feeds for Gold sentiment.
    Note: RSS feeds cannot be backtested historically, so this will only be used in live trading.
    """
    
    def __init__(self):
        self.feeds = [
            "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
            "https://www.kitco.com/rss/",
            "https://www.investing.com/rss/news_25.rss"
        ]
        
        self.bullish_keywords = [
            "gold rises", "gold rally", "safe haven",
            "inflation", "rate cut", "geopolitical",
            "war", "crisis", "recession fear"
        ]
        
        self.bearish_keywords = [
            "gold falls", "gold drops", "rate hike",
            "dollar strength", "risk on", "gold selloff"
        ]

    def fetch_and_analyze(self) -> Tuple[float, int, List[str]]:
        """
        Fetches live RSS feeds and computes a sentiment score based on keyword matching.
        Returns: (sentiment_score, news_count, top_headlines)
        """
        headlines = []
        
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]: # Take top 10 from each feed
                    title = entry.title.lower()
                    headlines.append(entry.title)
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {url}: {e}")
                
        if not headlines:
            return 0.0, 0, []
            
        bull_count = 0
        bear_count = 0
        
        for title in headlines:
            title_lower = title.lower()
            
            for word in self.bullish_keywords:
                if word in title_lower:
                    bull_count += 1
                    
            for word in self.bearish_keywords:
                if word in title_lower:
                    bear_count += 1
                    
        total_signals = bull_count + bear_count
        if total_signals == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (bull_count - bear_count) / total_signals
            
        top_headlines = headlines[:3] if len(headlines) >= 3 else headlines
        
        return sentiment_score, len(headlines), top_headlines

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sa = SentimentAnalyzer()
    score, count, top = sa.fetch_and_analyze()
    print(f"Score: {score}, Count: {count}")
    print("Top Headlines:", top)
