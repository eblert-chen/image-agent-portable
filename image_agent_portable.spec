# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Image Generation Agent — Portable Edition.

Build:  pyinstaller image_agent_portable.spec
Output: dist/image-agent-portable/
"""

import sys
from pathlib import Path

_root = Path(SPECPATH)

a = Analysis(
    [str(_root / 'app.py')],
    pathex=[str(_root)],
    binaries=[],
    datas=[
        (str(_root / 'static' / 'index.html'), 'static'),
    ],
    hiddenimports=[
        # LangChain / LangGraph internals
        'langgraph',
        'langgraph.graph',
        'langgraph.checkpoint',
        'langgraph.checkpoint.memory',
        'langgraph.pregel',
        'langgraph.channels',
        'langchain',
        'langchain_core',
        'langchain_openai',
        # FastAPI / Starlette / Uvicorn
        'uvicorn',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'starlette',
        'fastapi',
        # Database
        'aiosqlite',
        # HTTP
        'httpx',
        'aiofiles',
        # Image
        'PIL',
        # Utilities
        'tenacity',
        'multipart',
        'pydantic',
        'pydantic_settings',
        # Services — ensure all cloud model modules are included
        'services',
        'services.sdxl',
        'services.anime',
        'services.doubao',
        'services.face3d',
        'services.glm',
        'services.hunyuan',
        'services.llm',
        'services.modelscope',
        'services.pollinations',
        'services.qianfan',
        'services.runninghub',
        'services.scorer',
        'services.spark',
        'services.stability',
        'services.wan26',
        # Nodes
        'nodes',
        'nodes.prompt',
        'nodes.router',
        'nodes.generator',
        'nodes.critic',
        'nodes.decision',
        # Other modules
        'config',
        'state',
        'stats',
        'db',
        'key_manager',
        'graph',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'open_clip',
        'diffusers',
        'accelerate',
        'safetensors',
        'transformers',
        'tokenizers',
        'sympy',
        'mpmath',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='image-agent-portable',
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='image-agent-portable',
)
