"""Regression tests for github.com/wfzyx/von/issues/16 (jevcompat conformance report).

Three gaps found by mandu5's jevcompat suite against TypeSafe's /v1/systemone
contract:
  1. Structured `instructions` (object/array) were rejected with a 422.
  2. `confidence` used a raw top1-minus-top2 margin instead of TypeSafe's
     (n * p_max - 1) / (n - 1).
  3. `"questions": {}` returned 200 with empty answers instead of a 422
     (OpenAPI declares minProperties: 1).
"""

import pytest
from fastapi.testclient import TestClient

from von.backends.option_marker_backend import _margin_confidence
from von.server import app
from von.types import Choice, Noul, Score

client = TestClient(app)


def test_noul_accepts_object_instructions():
    q = Noul(instructions={"task": "Does the customer want money back?"})
    assert q.instructions == '{"task": "Does the customer want money back?"}'


def test_choice_accepts_array_instructions():
    q = Choice(instructions=["step one", "step two"], criteria={"a": None})
    assert q.instructions == '["step one", "step two"]'


def test_score_accepts_object_instructions():
    q = Score(instructions={"scale": "1-5"}, criteria=["low", "high"])
    assert q.instructions == '{"scale": "1-5"}'


def test_string_instructions_pass_through_unchanged():
    q = Noul(instructions="Does the customer want money back?")
    assert q.instructions == "Does the customer want money back?"


def test_server_accepts_structured_instructions():
    r = client.post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "refund please",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": {"task": "Does the customer want money back?"},
                }
            },
        },
    )
    assert r.status_code == 200


def test_empty_questions_dict_is_rejected():
    r = client.post(
        "/v1/systemone",
        json={"model": "jev-latest", "state": "x", "questions": {}},
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([0.286, 0.363, 0.351], 0.044),  # the issue's exact repro numbers
        ([0.5, 0.5], 0.0),  # n=2 at chance
        ([1.0, 0.0], 1.0),  # n=2 fully certain
        ([1.0], 1.0),  # single option, nothing to be uncertain against
        ([], 1.0),
    ],
)
def test_margin_confidence_matches_typesafe_formula(probs, expected):
    assert _margin_confidence(probs) == pytest.approx(expected, abs=1e-3)


def test_margin_confidence_equals_raw_margin_at_n_equals_2():
    # (n*p_max - 1)/(n-1) reduces exactly to top1-top2 margin only at n=2.
    probs = [0.7, 0.3]
    assert _margin_confidence(probs) == pytest.approx(probs[0] - probs[1], abs=1e-6)
