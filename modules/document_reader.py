# # Redundant - legacy - remove
#
# def summarize_pdf(self, path: str = None) -> str:
#     """Summarize most recent PDF or specified path."""
#     import glob
#     import os
#     from pypdf import PdfReader
#
#     if not path:
#         # find most recent PDF
#         search_paths = [
#             os.path.expanduser("~/Downloads/*.pdf"),
#             os.path.expanduser("~/Documents/*.pdf"),
#             os.path.expanduser("~/Desktop/*.pdf"),
#         ]
#         all_pdfs = []
#         for pattern in search_paths:
#             all_pdfs.extend(glob.glob(pattern))
#
#         if not all_pdfs:
#             return "No PDF files found, sir."
#
#         path = max(all_pdfs, key=os.path.getmtime)
#         print(f"[Vision] Latest PDF: {path}")
#
#     try:
#         reader = PdfReader(path)
#         text = ""
#         for page in reader.pages[:10]:  # first 10 pages max
#             text += page.extract_text() or ""
#
#         if not text.strip():
#             return "Couldn't extract text from that PDF, sir."
#
#         print(f"[Vision] Extracted {len(text)} chars from PDF")
#         return text
#
#     except Exception as e:
#         print(f"[Vision] PDF error: {e}")
#         return f"Error reading PDF, sir. {e}"