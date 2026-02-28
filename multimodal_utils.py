import os
import pdfplumber
from PIL import Image
import io

def extract_images_from_pdf(pdf_path, output_dir="extracted_images"):
    """
    Extracts all images from a PDF file and saves them to the output directory.
    Returns a list of paths to the saved images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # pdfplumber extracts images as a list of dictionaries
            for j, image_dict in enumerate(page.images):
                try:
                    # Get the bounding box of the image
                    bbox = (image_dict['x0'], image_dict['top'], image_dict['x1'], image_dict['bottom'])
                    
                    # Crop the page to the image's bounding box and convert to a PIL image
                    # This is often more reliable than extracting the raw stream
                    page_image = page.within_bbox(bbox).to_image(resolution=300)
                    
                    image_filename = f"page_{i+1}_img_{j+1}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    page_image.save(image_path)
                    image_paths.append(image_path)
                    print(f"Saved image: {image_path}")
                except Exception as e:
                    print(f"Error extracting image {j} from page {i}: {e}")
                    
    return image_paths

if __name__ == "__main__":
    # Test with the AWS overview PDF
    pdf_file = "data/aws-overview.pdf"
    if os.path.exists(pdf_file):
        extract_images_from_pdf(pdf_file)
    else:
        print(f"File {pdf_file} not found.")
