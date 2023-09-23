import requests
from bs4 import BeautifulSoup

def scrape_blog(url: str) -> str:
    """Simple HTML scraper that always returns something or an explicit error."""
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if res.status_code != 200:
            return f"❌ HTTP error {res.status_code}"
        soup = BeautifulSoup(res.text, "html.parser")
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
        if not text:
            text = " ".join(div.get_text(strip=True) for div in soup.find_all("div"))
        if not text:
            return "❌ No visible text found on page (site may use JavaScript)."
        return text[:8000]
    except Exception as e:
        return f"❌ Exception while scraping: {e}"
