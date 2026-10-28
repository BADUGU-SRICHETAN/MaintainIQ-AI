.PHONY: install data test eval eval-answers lint app docker

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

data:
	python scripts/generate_data.py

test:
	python -m pytest

eval:
	python scripts/evaluate.py

eval-answers:
	python scripts/evaluate_answers.py

lint:
	ruff check src tests scripts app

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t maintainiq-ai .
