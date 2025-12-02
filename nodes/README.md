# Nodes Directory

This directory contains all Gemini-related custom nodes for ComfyUI.

## Current Nodes

### JM Gemini Image Generator (`jm_gemini_node.py`)
- Text-to-image generation
- Image-to-image generation
- Image editing
- Support for multiple Gemini models

## Adding New Nodes

To add a new Gemini-related node:

1. Create a new Python file in this directory (e.g., `jm_gemini_video_node.py`)

2. Implement your node class with the standard ComfyUI node structure:

```python
class YourNewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # your inputs
            }
        }

    RETURN_TYPES = ("TYPE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "your_function"
    CATEGORY = "JM-Gemini"

    def your_function(self, ...):
        # your implementation
        pass

# Export mappings
NODE_CLASS_MAPPINGS = {
    "YourNewNode": YourNewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YourNewNode": "Your Display Name"
}
```

3. Update `nodes/__init__.py` to import your new node:

```python
from .jm_gemini_node import NODE_CLASS_MAPPINGS as IMAGE_MAPPINGS
from .jm_gemini_node import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_MAPPINGS
from .your_new_node import NODE_CLASS_MAPPINGS as YOUR_MAPPINGS
from .your_new_node import NODE_DISPLAY_NAME_MAPPINGS as YOUR_DISPLAY_MAPPINGS

# Merge all mappings
NODE_CLASS_MAPPINGS = {
    **IMAGE_MAPPINGS,
    **YOUR_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **IMAGE_DISPLAY_MAPPINGS,
    **YOUR_DISPLAY_MAPPINGS
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

4. Restart ComfyUI to load the new node

## Node Naming Convention

- File name: `jm_gemini_<feature>_node.py`
- Class name: `JMGemini<Feature><Type>`
- Category: `JM-Gemini` (consistent across all nodes)
- Display name: `JM Gemini <Feature> <Type>`

Examples:
- `jm_gemini_image_node.py` → `JMGeminiImageGenerator` → "JM Gemini Image Generator"
- `jm_gemini_video_node.py` → `JMGeminiVideoGenerator` → "JM Gemini Video Generator"
- `jm_gemini_chat_node.py` → `JMGeminiChatProcessor` → "JM Gemini Chat Processor"
