import glob
import os
import re

import pdfplumber

###################################################################################################################
# Core functions
###################################################################################################################


def is_table_of_contents(page_text, threshold=3):
    toc_keywords = ["item", "page", "contents", "part"]
    toc_count = sum(
        1 for keyword in toc_keywords if keyword.lower() in page_text.lower()
    )
    return toc_count >= threshold


def save_text_to_file(text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def extract_risk_factors_text(
    pdf_path, pdf_filename, min_length=50, toc_pages_to_check=4
):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                if page_num < toc_pages_to_check and is_table_of_contents(page_text):
                    continue
                text += page_text + "\n"

    item_1a_pattern = re.compile(
        r"(Item\s*(1A|IA|I\.A|1\.A)\s*[\.\-\:\|\—–]?[\.\-\:]?\s*\n?\s*Risk\s*Factors[\.\-\|\:\—–]?)|"
        r"(^Risk\s*Factors$)|"
        r"(^1A[\.\-\:]?\s*\n?\s*Risk\s*Factors)|"
        r"(Risk\s*Factors\s*1A[\.\-\:]?)",
        re.IGNORECASE | re.MULTILINE,
    )

    item_1b_pattern = re.compile(
        r"(Item\s*(1B|IB|I\.B|1\.B)\s*[\.\-\:\|\—–]?[\.\-\:]?\s*\n?\s*Unresolved\s*Staff\s*Comments[\.\-\|\:\—–]?)|"
        r"(^Unresolved\s*Staff\s*Comments$)|"
        r"(^1B[\.\-\:]?\s*\n?\s*Unresolved\s*Staff\s*Comments)|"
        r"(Unresolved\s*Staff\s*Comments\s*1B[\.\-\:]?)",
        re.IGNORECASE | re.MULTILINE,
    )

    item_2_pattern = re.compile(
        r"(Item\s*2\s*[\.\-\:\|\—–]?[\.\-\:]?\s*\n?\s*Properties[\.\-\|\:\—–]?)|"
        r"(^Properties$)",
        re.IGNORECASE | re.MULTILINE,
    )

    item_1a_match = item_1a_pattern.search(text)
    item_1b_match = item_1b_pattern.search(text)

    if not item_1a_match:
        print(f"[-] {pdf_filename}: Section 'Item 1A. Risk Factors' not found.")
        return None, text

    if not item_1b_match:
        item_1b_match = item_2_pattern.search(text)
        if not item_1b_match:
            print(f"[-] {pdf_filename}: Neither 'Item 1B' nor 'Item 2' found.")
            return None, text

    start_pos = item_1a_match.end()
    end_pos = item_1b_match.start()
    section_text = text[start_pos:end_pos].strip()

    if len(section_text) >= min_length:
        print(f"[+] {pdf_filename}: Risk section extracted successfully.")
    else:
        print(f"[!] {pdf_filename}: Extracted, but section is too short.")

    return section_text, text


###################################################################################################################
# DVC Pipeline Execution
###################################################################################################################
if __name__ == "__main__":
    PDF_DIR = "data/pdf"
    FULL_TXT_DIR = "data/full-txt"
    RISK_TXT_DIR = "data/risk-txt"

    print("Starting text extraction...")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        section_text, full_text = extract_risk_factors_text(pdf_path, filename)

        # Save full text
        save_text_to_file(
            full_text, os.path.join(FULL_TXT_DIR, filename.replace(".pdf", ".txt"))
        )

        # Save risk section if found
        if section_text:
            save_text_to_file(
                section_text,
                os.path.join(RISK_TXT_DIR, filename.replace(".pdf", ".txt")),
            )

    print("Extraction completed.")
