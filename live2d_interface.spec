# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['live2d_interface.py'],
    pathex=['E:/CosyVoice'],
    binaries=[],
    datas=[
        ('configs/config.json', 'configs'),
        ('rss_data/rss_data.json', 'rss_data'),
        ('models', 'models'),  
        ('UI_icons', 'UI_icons'),
        ('whisper_model', 'whisper_model'),
        ('pretrained_models', 'pretrained_models'),
        ('cosyvoice', 'cosyvoice')
    ],
    hiddenimports=[],
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
    [],
    exclude_binaries=True,
    name='OwnNeuro',
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
    icon=['logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='live2d_interface',
)
