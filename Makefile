.PHONY: setup pull features research model backtest test lint app week help

# Homebrew Python on a Mac will refuse `pip install` into the system
# (PEP 668). We keep this project's packages in .venv instead.
VENV := .venv
PYTHON := $(VENV)/bin/python
SYS_PYTHON ?= python3

help:
	@echo "First time on this machine:"
	@echo "  make setup      create .venv and install packages"
	@echo "  make app        open http://127.0.0.1:8501 in your browser"
	@echo "First-time data build is a button on the Build data page."
	@echo "Optional CLI: make pull / features / research / model / backtest"

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
	@echo "Next:  make app"

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
	@echo ""
	@echo "Starting the web app at  http://127.0.0.1:8501"
	@echo "Keep this terminal open. The site is in your browser."
	@echo ""
	$(PYTHON) launch_dashboard.py

week: $(PYTHON)
	$(PYTHON) run_week.py

all: pull features research model backtest
