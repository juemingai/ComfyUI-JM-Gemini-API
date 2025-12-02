# ComfyUI-JM-Gemini-API

A custom node for ComfyUI that generates images using Google's Gemini API, supporting both text-to-image and image-to-image generation.

English | [简体中文](README_CN.md)

## Features

- Support for multiple Gemini models:
  - `gemini-3-pro-image-preview` (default, with 2K resolution)
  - `gemini-2.5-flash-image`
- Text-to-image generation
- Image-to-image generation (single or multiple input images)
- Image editing mode (single image input)
- Configurable aspect ratios (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
- Resolution control (1K, 2K, 4K) - only for gemini-3-pro-image-preview
- Support up to 10 input images
- Automatic image saving to ComfyUI output directory

## Project Structure

```
ComfyUI-JM-Gemini-API/
├── __init__.py              # Main entry point for ComfyUI
├── nodes/                   # Node implementations directory
│   ├── __init__.py         # Nodes package initializer
│   └── jm_gemini_node.py   # Gemini image generator node
├── requirements.txt         # Python dependencies
├── README.md               # Documentation
└── .gitignore             # Git ignore rules
```

This modular structure makes it easy to add more Gemini-related nodes in the future. Simply add new node files to the `nodes/` directory and import them in `nodes/__init__.py`.

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-JM-Gemini-API.git
```

2. Install required dependencies:

```bash
cd ComfyUI-JM-Gemini-API
pip install -r requirements.txt
```

3. Restart ComfyUI

## Requirements

- Python 3.8+
- ComfyUI
- google-genai >= 0.2.0
- Pillow >= 10.0.0
- torch
- torchvision
- numpy

## Usage

### Getting Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for use in the node

### Node Parameters

#### Required Inputs:

- **gemini_api_key**: Your Gemini API key (string)
- **prompt**: Text prompt describing the image you want to generate (multiline text)
- **model**: Choose between:
  - `gemini-3-pro-image-preview` (default, supports 1K/2K/4K resolution)
  - `gemini-2.5-flash-image` (faster, aspect ratio only)
- **aspect_ratio**: Image aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9)
- **resolution**: Image resolution (1K, 2K, 4K)
  - Note: Only effective for `gemini-3-pro-image-preview` model
  - Default: 2K

#### Optional Inputs:

- **image1 ~ image10**: Up to 10 optional image inputs for image-to-image generation
  - Connect output from Load Image node
  - Can be left empty for text-to-image generation

#### Outputs:

- **image**: Generated image (ComfyUI IMAGE tensor format)
  - Can be connected to Preview Image or Save Image nodes
  - Automatically saved to ComfyUI output directory

### Usage Examples

#### 1. Text-to-Image Generation

1. Add "JM Gemini Image Generator" node to your workflow
2. Enter your Gemini API key
3. Write your prompt
4. Select model and aspect ratio
5. Leave all image inputs empty
6. Connect output to Preview Image node
7. Run the workflow

#### 2. Image Editing (Single Image)

1. Add Load Image node and load your image
2. Add "JM Gemini Image Generator" node
3. Connect Load Image output to image1 input
4. Enter editing instructions in prompt (e.g., "Add a sunset background")
5. Configure model and parameters
6. Run the workflow

#### 3. Image-to-Image (Multiple Images)

1. Add multiple Load Image nodes
2. Connect them to image1, image2, etc.
3. Enter a prompt describing how to combine/transform the images
4. Run the workflow

## Model Differences

### gemini-3-pro-image-preview
- Supports resolution parameter (1K, 2K, 4K)
- Default resolution: 2K
- Supports single image editing
- Higher quality output

### gemini-2.5-flash-image
- Faster generation
- Uses aspect ratio only (no resolution parameter)
- Good for quick iterations

## Troubleshooting

### Common Issues

1. **"Gemini API key is required"**
   - Make sure you've entered a valid API key

2. **"No images were generated"**
   - Check your prompt is clear and descriptive
   - Try a different aspect ratio or resolution
   - Verify your API key is valid and has sufficient quota

3. **"Failed to generate image"**
   - Check your internet connection
   - Verify API key permissions
   - Check Gemini API service status

4. **Image quality issues**
   - For gemini-3-pro-image-preview, try using different resolutions (1K, 2K, or 4K)
   - Make your prompt more detailed and specific

## Output Directory

Generated images are automatically saved to:
- `ComfyUI/output/` directory
- Filename format: `{model}_{mode}_{timestamp}.png`
  - Example: `gemini3pro_text2img_1234567890.png`

## License

MIT License

## Credits

Developed by JM

Based on Google's Gemini API

## Support

For issues and feature requests, please visit the [GitHub repository](https://github.com/yourusername/ComfyUI-JM-Gemini-API/issues)

## Changelog

### Version 1.0.0
- Initial release
- Support for gemini-3-pro-image-preview and gemini-2.5-flash-image
- Text-to-image generation
- Image-to-image generation (up to 10 images)
- Configurable aspect ratios and resolutions
