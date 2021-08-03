VENV = venv
ifeq ($(OS),Windows_NT)
	VENV_BIN = Scripts
else
	VENV_BIN = bin
endif

.PHONY: run
run: $(VENV)/$(VENV_BIN)/activate
	./$(VENV)/$(VENV_BIN)/python src/se.py

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf build

$(VENV)/$(VENV_BIN)/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/$(VENV_BIN)/pip install -r requirements.txt

.PHONY: sphinx
sphinx: $(VENV)/$(VENV_BIN)/activate
	./$(VENV)/$(VENV_BIN)/sphinx-build -b html docs build

.PHONY: help
help:
	@echo "make run:    Run the project"
	@echo "make clean:  Clean the project"
	@echo "make sphinx: Build documentation"
