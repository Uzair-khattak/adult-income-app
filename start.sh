#!/bin/bash
uvicorn main:app --host 127.0.0.1 --port 8000 &
sleep 6
streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --server.headless true
