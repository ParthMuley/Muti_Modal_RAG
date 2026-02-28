import pdfplumber

def find_shared_responsibility_pages(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        results = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and "Shared Responsibility" in text:
                count = text.count("Shared Responsibility")
                results.append((i+1, count, text[:200]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        for page_num, count, excerpt in results:
            print(f"Page {page_num}: {count} occurrences. Excerpt: {excerpt}")

if __name__ == "__main__":
    find_shared_responsibility_pages("data/aws-overview.pdf")
