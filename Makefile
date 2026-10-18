.PHONY: test lint kaggle-zip

test:
	python -m pytest -q

lint:
	ruff check .
	ruff format --check .

kaggle-zip:
	mkdir -p dist
	git archive --format=zip -o dist/cuad-encoder.zip HEAD cuad_encoder pyproject.toml
