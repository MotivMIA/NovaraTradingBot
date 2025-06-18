# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['features/dashboard.py'],
    pathex=[],
    binaries=[],
    datas=[('features', 'features'), ('/Users/nathanwilliams/Documents/projects/NovaraTradingBot/venv/lib/python3.11/site-packages/streamlit-1.39.0.dist-info', 'streamlit-1.39.0.dist-info')],
    hiddenimports=['streamlit', 'streamlit.runtime', 'streamlit.version', 'importlib_metadata', 'importlib.metadata'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
