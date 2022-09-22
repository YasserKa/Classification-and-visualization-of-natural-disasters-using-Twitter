.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install: 
	@echo "Installing..."
	pipenv install
	pipenv run pre-commit install
	python -m spacy download en_core_web_sm

activate:
	@echo "Activating virtual environment"
	pipenv shell

initialize_git:
	git init 

pull_data:
	pipenv run dvc pull

setup: initialize_git install

test:
	pytest
 
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
