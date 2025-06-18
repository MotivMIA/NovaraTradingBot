#!/bin/bash
source venv/bin/activate
STREAMLIT_DIST_INFO=$(ls -d $(python -c 'import site; print(site.getsitepackages()[0])')/streamlit-*.dist-info 2>/dev/null | head -n 1)
if [ -z "$STREAMLIT_DIST_INFO" ]; then
    echo "Error: Streamlit dist-info directory not found"
    exit 1
fi
pyinstaller --onefile \
    --add-data 'features:features' \
    --add-data "${STREAMLIT_DIST_INFO}:$(basename ${STREAMLIT_DIST_INFO})" \
    --hidden-import streamlit \
    --hidden-import streamlit.runtime \
    --hidden-import streamlit.version \
    --hidden-import importlib_metadata \
    --hidden-import importlib.metadata \
    features/dashboard.py