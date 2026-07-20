import os

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}...")
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(text if text else "")
                f.write("\n")
        print("Success using pypdf")
        return True
    except Exception as e:
        print(f"pypdf failed: {e}")
    
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(text if text else "")
                f.write("\n")
        print("Success using PyPDF2")
        return True
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            with open(txt_path, "w", encoding="utf-8") as f:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    f.write(f"--- PAGE {i+1} ---\n")
                    f.write(text if text else "")
                    f.write("\n")
        print("Success using pdfplumber")
        return True
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(doc):
                text = page.get_text()
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(text if text else "")
                f.write("\n")
        print("Success using PyMuPDF (fitz)")
        return True
    except Exception as e:
        print(f"PyMuPDF failed: {e}")

    return False

if __name__ == "__main__":
    pdf_dir = "/home/blybq/code-project/cs336/lectures"
    for num in ["08", "09", "11"]:
        pdf_path = os.path.join(pdf_dir, f"lecture_{num}.pdf")
        txt_path = os.path.join(pdf_dir, f"lecture_{num}_extracted.txt")
        if os.path.exists(pdf_path):
            success = extract_pdf_text(pdf_path, txt_path)
            if not success:
                print(f"Could not extract text from {pdf_path}")
        else:
            print(f"{pdf_path} does not exist")
