import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import extract_claims, classify_nli_result


def test_extract_claims():
    text = (
        "Paris is the capital of France. "
        "It is located in Europe. "
        "The city has a long history."
    )

    claims = extract_claims(text)

    assert len(claims) == 3
    assert claims[0] == "Paris is the capital of France."
    assert claims[1] == "It is located in Europe."
    assert claims[2] == "The city has a long history."


def test_extract_claims_empty():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_supported_decision():
    result = classify_nli_result(
        {
            "probabilities": {
                "contradiction": 0.001,
                "entailment": 0.990,
                "neutral": 0.009,
            }
        }
    )

    assert result["label"] == "SUPPORTED"


def test_contradicted_decision():
    result = classify_nli_result(
        {
            "probabilities": {
                "contradiction": 0.990,
                "entailment": 0.001,
                "neutral": 0.009,
            }
        }
    )

    assert result["label"] == "CONTRADICTED"


def test_unknown_when_neutral_is_highest():
    result = classify_nli_result(
        {
            "probabilities": {
                "contradiction": 0.02,
                "entailment": 0.05,
                "neutral": 0.93,
            }
        }
    )

    assert result["label"] == "UNKNOWN"


def test_unknown_when_margin_is_small():
    result = classify_nli_result(
        {
            "probabilities": {
                "contradiction": 0.24,
                "entailment": 0.40,
                "neutral": 0.36,
            }
        }
    )

    assert result["label"] == "UNKNOWN"


def test_probability_values_are_preserved():
    probabilities = {
        "contradiction": 0.10,
        "entailment": 0.80,
        "neutral": 0.10,
    }

    result = classify_nli_result(
        {
            "probabilities": probabilities
        }
    )

    assert result["probabilities"] == probabilities
    assert result["label"] == "SUPPORTED"