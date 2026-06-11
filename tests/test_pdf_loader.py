from src.ingestion.pdf_loader import PDFLoader


loader = PDFLoader("data/pdfs/Introduction_to_Machine_Learning.pdf")
text = loader.extract_text()

print(text[:1000])
print(f"\nTotal Characters: {len(text)}")