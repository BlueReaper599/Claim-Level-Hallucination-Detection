import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import HallucinationDetectionPipeline


@pytest.fixture(scope="module")
def pipeline():
    """
    Load the complete verification pipeline once for this test module.

    Loading the retrieval models and NLI model is expensive, so all
    integration tests share the same pipeline instance.
    """
    return HallucinationDetectionPipeline()


def test_supported_claim(pipeline):
    result = pipeline.verify_claim(
        "Paris is the capital of France."
    )

    assert result["label"] == "SUPPORTED"
    assert result["evidence"] is not None
    assert result["confidence"] > 0.50


def test_contradicted_claim(pipeline):
    result = pipeline.verify_claim(
        "Paris is the capital of Germany."
    )

    assert result["label"] == "CONTRADICTED"
    assert result["evidence"] is not None
    assert result["confidence"] > 0.50


def test_unknown_claim(pipeline):
    result = pipeline.verify_claim(
        "Paris has exactly 10 million inhabitants."
    )

    assert result["label"] == "UNKNOWN"
    assert result["evidence"] is not None


def test_answer_level_pipeline(pipeline):
    answer = (
        "Paris is the capital of France. "
        "Paris is the capital of Germany."
    )

    result = pipeline.verify_answer(answer)

    assert result["summary"]["total_claims"] == 2

    assert result["summary"]["supported"] >= 1
    assert result["summary"]["contradicted"] >= 1

    assert len(result["claims"]) == 2


def test_empty_claim_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.verify_claim("")


def test_empty_answer_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.verify_answer("")