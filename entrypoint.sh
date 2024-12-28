#!/bin/bash
set -e

# Start Streamlit with xvfb-run
xvfb-run -a --server-args="-screen 0 1024x768x24" streamlit run gui/gui_streamlit.py --server.headless=true --server.address=0.0.0.0 --server.port=8501
