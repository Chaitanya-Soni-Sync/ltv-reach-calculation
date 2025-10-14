#!/bin/bash
# Convenience script to run the LTV Reach Calculation Pipeline

# Activate virtual environment
source venv/bin/activate

# Run the main pipeline
python modules/main.py

# Deactivate virtual environment
deactivate

