import pdfplumber
import requests
from bs4 import BeautifulSoup
import os
import json

def extract_links_from_pdf(pdf_path):
    """Finds all unique external URLs in the PDF."""
    links = set()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # pdfplumber extracts hyperlinks in the 'annots' (annotations)
                if page.annots:
                    for annot in page.annots:
                        # Some versions of pdfplumber use different keys
                        uri = None
                        if 'uri' in annot:
                            uri = annot['uri']
                        elif 'URI' in annot:
                            uri = annot['URI']
                        
                        if uri and isinstance(uri, str) and uri.startswith("http"):
                            links.add(uri)
    except Exception as e:
        print(f"Error extracting links from PDF: {e}")
    return list(links)

def scrape_url_content(url):
    """Fetches the main text from a webpage and returns it as a string."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Simple extraction: remove scripts and styles, then get text
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text_content[:15000] # Increased limit to 15k chars
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

if __name__ == "__main__":
    pdf_file = "data/aws-overview.pdf"
    if os.path.exists(pdf_file):
        print(f"Scanning {pdf_file} for links...")
        found_links = extract_links_from_pdf(pdf_file)
        print(f"Found {len(found_links)} unique links.")
        
        # Save links to a file to show you
        with open("found_links.json", "w") as f:
            json.dump(found_links, f, indent=4)
        
        print("\nFirst 10 Links Found:")
        for link in found_links[:10]:
            print(f"- {link}")
    else:
        print(f"File {pdf_file} not found.")
