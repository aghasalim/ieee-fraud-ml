.PHONY: setup data eda validate features train app docker test clean
PY := .venv/bin/python

setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

data:            ## download from Kaggle (needs ~/.kaggle/kaggle.json + accepted rules)
	$(PY) -m src.fraud.data

eda:             ## missingness, imbalance, temporal structure -> reports/
	$(PY) -m src.fraud.eda

validate:        ## the headline experiment on synthetic data (no credentials)
	$(PY) -m src.fraud.experiments.validation_gap

leakage-real: data   ## the same 2x2 on all 590k real transactions
	$(PY) -m src.fraud.experiments.leakage_real

train:           ## feature ablation + overfitting report (honest split)
	$(PY) -m src.fraud.train

train-final:     ## fit on all data and save artifacts/model.pkl for the app
	$(PY) -m src.fraud.train --final

error-analysis: ## segments, review budget, calibration, missed-fraud profile
	$(PY) -m src.fraud.error_analysis

app:
	.venv/bin/streamlit run app/streamlit_app.py

docker:
	docker build -t ieee-fraud-ml .

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/processed artifacts/*.pkl reports/figures
