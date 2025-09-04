#!/bin/bash
python dsview/obsidian/setup.py

fastapi run dsview/api.py --host 0.0.0.0 & \
streamlit run dsview/interface/dsview_dashboard.py --server.address 0.0.0.0
marimo run dsview/interface/dsview_dashboard.py --host 0.0.0.0 -p 8080
