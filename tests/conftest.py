import json
from pathlib import Path

import pytest

CONTEXT = (
    "This Agreement is made between Acme Corp and Beta LLC. "
    "This Agreement shall be governed by the laws of the State of Delaware. "
    "Neither party may assign this Agreement without consent."
)


def question(clause_type: str) -> str:
    return (
        f'Highlight the parts (if any) of this contract related to "{clause_type}" '
        "that should be reviewed by a lawyer."
    )


def answer(text: str) -> dict:
    return {"text": text, "answer_start": CONTEXT.index(text)}


@pytest.fixture
def cuad_json(tmp_path: Path) -> Path:
    """Two contracts in the shape of CUAD's test.json."""
    qas = [
        {
            "id": "law",
            "question": question("Governing Law"),
            "answers": [answer("governed by the laws of the State of Delaware")],
        },
        {
            "id": "parties",
            "question": question("Parties"),
            "answers": [answer("Acme Corp"), answer("Beta LLC")],
        },
        {"id": "assign", "question": question("Anti-Assignment"), "answers": []},
    ]
    data = [
        {"title": title, "paragraphs": [{"context": CONTEXT, "qas": qas}]}
        for title in ("contract-a", "contract-b")
    ]
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps({"version": "test", "data": data}))
    return path
