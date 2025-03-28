# Vercel Configuration for CSV Dataset Cleaning Assistant
# This file configures the application for deployment on Vercel

from pathlib import Path
import sys
import os

# Add the current directory to the path so that local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Streamlit app
from app import main

# Vercel requires a wsgi.py file with an app variable
app = main
