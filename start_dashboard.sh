#!/bin/bash
# Start the LTV Reach Calculator Dashboard

echo "🚀 Starting LTV Reach Calculator Dashboard..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py

# Deactivate when done
deactivate

