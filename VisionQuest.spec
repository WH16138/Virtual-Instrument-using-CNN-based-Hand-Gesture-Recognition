# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

hiddenimports = [
    'mediapipe.tasks.python.vision.hand_landmarker',
    'mediapipe.tasks.python.vision.core.image',
    'sklearn.pipeline',
    'sklearn.preprocessing._data',
    'sklearn.neural_network._multilayer_perceptron',
    'sklearn.neural_network._base',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('models', 'models'),
        ('web', 'web'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'torch',
        'jax',
        'dask',
        'sentencepiece',
        'matplotlib',
        'pandas',
        'pyarrow',
        'numba',
        'llvmlite',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionQuest',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionQuest',
)
