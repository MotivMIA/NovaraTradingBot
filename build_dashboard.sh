#!/bin/bash
source venv/bin/activate
pyinstaller --onefile 

--add-data 'features:features' 

--add-data "$(python -c 'import site; print(site.getsitepackages()[0])')/streamlit-1.39.0.dist-info:streamlit-1.39.0.dist-info" 

--hidden-import streamlit 

--hidden-import streamlit.runtime 

--hidden-import streamlit.version 

--hidden-import importlib_metadata 

--hidden-import importlib.metadata 

features/dashboard.py
