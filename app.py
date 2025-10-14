#!/usr/bin/env python3
"""
LTV Reach Calculator - Streamlit Dashboard Entry Point
Main application file to launch the dashboard
"""

import sys
import os

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Import and run dashboard
from modules.dashboard import main

if __name__ == "__main__":
    main()

