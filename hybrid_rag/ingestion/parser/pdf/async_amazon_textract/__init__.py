"""Parser AWS Textract mode async (StartDocumentAnalysis + polling, tanpa SNS/SQS).

Hanya tahap PDF → raw JSON yang berbeda dari ``amazon_textract``; raw JSON-nya
berbentuk sama sehingga ``amazon_textract.reconstruct_content_list`` dipakai apa adanya.
"""
from .parse import parse_document_to_json

__all__ = ["parse_document_to_json"]
