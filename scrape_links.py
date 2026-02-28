import os
import json
from bs4 import BeautifulSoup
import requests

def scrape_url_content(url):
    """Simple scraper function."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()[:15000]
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if not os.path.exists("found_links.json"):
        print("Run link_extractor.py first.")
        return
        
    with open("found_links.json", "r") as f:
        links = json.load(f)
    
    # Priority list
    to_scrape = []
    # Shared Responsibility link
    for link in links:
        if "shared-responsibility-model" in link:
            to_scrape.append(link)
            break
    
    # Add a few others
    keywords = ["vpc", "ec2", "s3", "iam", "securityhub"]
    for kw in keywords:
        for link in links:
            if f"/{kw}/" in link.lower() and link not in to_scrape:
                to_scrape.append(link)
                break
    
    os.makedirs("./scraped_content", exist_ok=True)
    
    for i, url in enumerate(to_scrape):
        print(f"[{i+1}/{len(to_scrape)}] Scraping {url}...")
        content = scrape_url_content(url)
        if content:
            safe_name = url.split("//")[-1].replace("/", "_").replace(".", "_")[:100]
            with open(f"./scraped_content/{safe_name}.txt", "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n\n{content}")
            print(f"Saved {safe_name}.txt")

if __name__ == "__main__":
    main()
