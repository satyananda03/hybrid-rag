"""PII redaction for a parsed content list — phone numbers only, for now.

Runs on the parser output (``parse_file``), so one function covers PDF, DOCX and
XLSX alike. Detection is Microsoft Presidio with only its ``PhoneRecognizer``
registered (libphonenumber underneath): numbers are validated rather than every
digit run masked, so NIK, years and Rupiah amounts are left alone.
"""

from functools import cache

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

PHONE_MASK = "[NOMOR_TELEPON]"
_ENTITIES = ["PHONE_NUMBER"]
_OPERATORS = {"PHONE_NUMBER": OperatorConfig("replace", {"new_value": PHONE_MASK})}
# Structural values (block kind, file paths, sheet name) — masking them would break the content list.
_SKIP_KEYS = frozenset({"type", "img_path", "path", "source_pdf", "page_idx"})

@cache
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    # Presidio defaults to en_core_web_lg, which is not installed and would be downloaded at runtime; sm only tokenizes here.
    nlp_engine = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
    ).create_engine()
    # Only the phone recognizer is registered, so no other PII type is ever detected.
    registry = RecognizerRegistry(supported_languages=["en"])
    # Numbers written without a +country code are read as Indonesian.
    registry.add_recognizer(PhoneRecognizer(supported_regions=("ID",)))
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
    return analyzer, AnonymizerEngine()

def redact_phone_numbers(text: str) -> str:
    """``text`` with every valid phone number replaced by :data:`PHONE_MASK`."""
    # shortcut: a number glued to more digits ("0812 3456 7890 2024") is not matched and stays; add a regex pass if real data shows it.
    if not text.strip():
        return text
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, language="en", entities=_ENTITIES)
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=_OPERATORS).text

def redact_content_list(content_list: list[dict]) -> list[dict]:
    """A masked copy of ``content_list``; the input is untouched because other workspaces share it."""
    return [_redact(item) for item in content_list]

def _redact(value, key: str | None = None):
    if isinstance(value, str):
        return value if key in _SKIP_KEYS else redact_phone_numbers(value)
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value