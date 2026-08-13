.PHONY: setup data eda validate features train app docker test clean
PY := .venv/bin/python

setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

data:            ## download from Kaggle (needs ~/.kaggle/kaggle.json + accepted rules)
	$(PY) -m src.fraud.data

eda:             ## missingness, imbalance, temporal structure -> reports/
	$(PY) -m src.fraud.eda

validate:        ## the headline experiment: random KFold vs chronological CV
	$(PY) -m src.fraud.experiments.validation_gap

features:
	$(PY) -m src.fraud.features

train:
	$(PY) -m src.fraud.train

app:
	.venv/bin/streamlit run app/streamlit_app.py

docker:
	docker build -t ieee-fraud-ml .

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/processed artifacts/*.pkl reports/figures
