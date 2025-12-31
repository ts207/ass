MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := help

PYTHON ?= python3
APP_MODULE ?= app.chat

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run ARGS=\"--debug\"   # run the assistant CLI (for ds/life agents)"
	@echo "  make shell                  # start an interactive shell with PYTHONPATH set"

.PHONY: run
run:
	@PYTHONPATH=. $(PYTHON) -m $(APP_MODULE) $(ARGS)

.PHONY: shell
shell:
	@echo "Launching shell with PYTHONPATH=."
	PYTHONPATH=. $(SHELL)
