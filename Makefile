.PHONY: setup pull features research model backtest test lint app week

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt

pull:
	$(PYTHON) ingest/pull_nflverse.py

features:
	$(PYTHON) features/team_games.py
	$(PYTHON) features/compensated.py

research:
	$(PYTHON) research/stability_oddeven.py
	$(PYTHON) research/stability_firstn.py

model:
	$(PYTHON) model/priors.py
	$(PYTHON) model/weights.py
	$(PYTHON) model/spread.py

backtest:
	$(PYTHON) backtest/walk_forward.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

app:
	$(PYTHON) -m streamlit run app.py --server.headless true

week:
	$(PYTHON) run_week.py

all: pull features research model backtest
