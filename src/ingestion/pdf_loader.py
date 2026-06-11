import fitz


class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        document = fitz.open(self.pdf_path)

        all_text = []

        for page in document:
            page_text = page.get_text()
            all_text.append(page_text)
        
        total_pages = len(document)
        print(f"Pages: {total_pages}")  

        return "\n".join(all_text)
        