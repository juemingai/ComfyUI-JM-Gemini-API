"""
ComfyUI-JM-Gemini-API
A custom node for ComfyUI that generates images using Google's Gemini API

Author: JM
Version: 1.0.0
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 设置版本信息
__version__ = "1.0.0"

# 打印加载信息
print("\033[34m[JM-Gemini-API]\033[0m: Node loaded successfully")
print(f"\033[34m[JM-Gemini-API]\033[0m: Version {__version__}")
print("\033[34m[JM-Gemini-API]\033[0m: Support models: gemini-3-pro-image-preview, gemini-2.5-flash-image")
