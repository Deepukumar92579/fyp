import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter
import json
import io

# Specify the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text_from_pdf(pdf_path):                       
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        
        for page_number in range(num_pages):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            
            # Apply additional preprocessing
            img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarization
            img = img.filter(ImageFilter.SHARPEN)  # Sharpening
            
            # Save the image for debugging purposes (comment out in production)
            img.save(f"page_{page_number}.png")
            
            # Perform OCR
            extracted_text = pytesseract.image_to_string(img, lang='eng')
            print(f"Extracted text from page {page_number}: {extracted_text}")
            text += extracted_text
            
        pdf_document.close()
        return text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def parse_text_to_json(text):
    data = []
    if text:
        lines = text.splitlines()
        for line in lines:
            if '—' in line:
                parts = line.split('—')
                long_word = parts[0].strip()
                segments = parts[1].split()
                data.append({
                    "long_word": long_word,
                    "segments": segments
                })
    return data

def save_to_json(data, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Data saved to {output_file}')
    except Exception as e:
        print(f"Error saving to JSON: {e}")

pdf_path = 'tesrr.pdf'
output_file = 'output.json'

text = extract_text_from_pdf(pdf_path)
if text:
    data = parse_text_to_json(text)
    save_to_json(data, output_file)
else:
    print("No text extracted from the PDF.")
