import fitz

from pathlib import Path


PDF_PATH = Path(
    "data/documents/LLM -1  SYLLABUS.pdf"
)


pdf = fitz.open(PDF_PATH)

print(f"Pages: {len(pdf)}")

for page_number in [0, 1, 8]:

    page = pdf[page_number]

    text = page.get_text("text")

    print("\n")
    print("=" * 80)
    print(f"PAGE {page_number + 1}")
    print("=" * 80)

    print(text[:2000])


pdf.close()