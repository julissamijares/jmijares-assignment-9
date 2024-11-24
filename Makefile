# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Install dependencies
install:
	python -m venv $(VENV)
	$(VENV)\Scripts\pip install -r requirements.txt

# Run the Flask application
run:
	set FLASK_APP=$(FLASK_APP) && set FLASK_ENV=development && $(VENV)\Scripts\flask run --port 3000

# Clean up virtual environment
clean:
	rmdir /S /Q $(VENV)

# Reinstall all dependencies
reinstall: clean install
