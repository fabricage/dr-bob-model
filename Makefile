.PHONY: setup pull features research model backtest test lint app week help

# Homebrew Python on a Mac will refuse `pip install` into the system
# (PEP 668). We keep this project's packages in .venv instead.
VENV := .venv
PYTHON := $(VENV)/bin/python
SYS_PYTHON ?= python3

help:
	@echo "First time on this machine:"
	@echo "  make setup      create .venv and install packages"
	@echo "  make pull       download nflverse data"
	@echo "  make features   team-game stats + opponent adjustment"
	@echo "  make research   stability studies"
	@echo "  make model      priors, weights, sample board"
	@echo "  make backtest   walk-forward evaluation"
	@echo "  make app        open the Streamlit dashboard"
	@echo "Later, in season:  make week"

$(PYTHON):
	@echo "No virtual environment yet. Run this first:"
	@echo "  make setup"
	@exit 1

setup:
	$(SYS_PYTHON) -m venv $(VENV)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Setup complete. Packages live in $(VENV)/ (not in Homebrew Python)."
	@echo "Next:  make pull"

pull: $(PYTHON)
	$(PYTHON) ingest/pull_nflverse.py

features: $(PYTHON)
	$(PYTHON) features/team_games.py
	$(PYTHON) features/compensated.py

research: $(PYTHON)
	$(PYTHON) research/stability_oddeven.py
	$(PYTHON) research/stability_firstn.py

model: $(PYTHON)
	$(PYTHON) model/priors.py
	$(PYTHON) model/weights.py
	$(PYTHON) model/spread.py

backtest: $(PYTHON)
	$(PYTHON) backtest/walk_forward.py

test: $(PYTHON)
	$(PYTHON) -m pytest

lint: $(PYTHON)
	$(PYTHON) -m ruff check .

app: $(PYTHON)
	$(PYTHON) -m streamlit run app.py

week: $(PYTHON)
	$(PYTHON) run_week.py

all: pull features research model backtest
