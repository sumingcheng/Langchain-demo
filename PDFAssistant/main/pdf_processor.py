# pdf_processor.py
import fitz  # PyMuPDF


class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            texts = [page.get_text() for page in doc]
            doc.close()
            return texts
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return []
