"""The whole flow on one document: ``python3 -m hybrid_rag.ingestion.chunking.legal_chunker file.pdf``"""

from hybrid_rag.ingestion.chunking.legal_chunker.debug_input import load_content_list, pdf_argument, text_of
from hybrid_rag.ingestion.chunking.legal_chunker.pipeline import analyze_regulation, remove_regulation_headers

pdf = pdf_argument()
content_list = load_content_list(pdf)
cleaned = remove_regulation_headers(content_list)
content = text_of(cleaned)
regulation = analyze_regulation(content, pdf.name)

print(f"Blok dibuang: {len(content_list) - len(cleaned)}")
if regulation is None:
    print("Bukan pola peraturan: dokumen dipotong dengan cara biasa.")
else:
    print(f"Nama  : {regulation.name}")
    print(f"Pasal : {len(regulation.articles)}, {regulation.articles[0].title} .. {regulation.articles[-1].title}")
    print(f"Bolong: {[article.title for article in regulation.articles if ' - ' in article.title] or '-'}")
    print(f"Akhir : karakter {regulation.end} dari {len(content)}")
    print(f"Contoh: {regulation.citation(regulation.articles[0])}")
