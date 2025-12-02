"""
ComfyUI-JM-Gemini-API
A custom node for ComfyUI that generates images using Google's Gemini API
"""

import os
import time
import mimetypes
import io
import logging
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# 设置日志
logger = logging.getLogger(__name__)

# 常量定义
GEMINI_3_PRO_MODEL = "gemini-3-pro-image-preview"
GEMINI_2_5_FLASH_MODEL = "gemini-2.5-flash-image"

# 模型配置
MODEL_CONFIGS = {
    "gemini-3-pro-image-preview": {
        "supports_image_edit": True,
        "resolution_type": "image_size",
        "default_resolution": "2K"
    },
    "gemini-2.5-flash-image": {
        "supports_image_edit": True,
        "resolution_type": "aspect_ratio",
        "default_resolution": "1024x1024"
    }
}

# 分辨率映射 - Gemini 3 Pro支持的分辨率
RESOLUTION_MAPPING = {
    "480p": "1K",
    "720p": "1K",
    "768P": "2K",
    "1080p": "2K",
    "1080P": "2K",
    "2K": "2K",
    "4K": "4K"
}

# Gemini 2.5 Flash Image 支持的宽高比及对应分辨率
ASPECT_RATIO_RESOLUTIONS = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672"
}

DEFAULT_RESOLUTION = "2K"


def tensor2pil(image_tensor):
    """
    将ComfyUI的tensor格式图像转换为PIL Image
    ComfyUI图像格式: (batch, height, width, channels) 值范围0-1
    """
    # 确保是4D tensor
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # 转换为numpy并调整值范围到0-255
    image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    # 转换为PIL Image
    return Image.fromarray(image_np)


def pil2tensor(image):
    """
    将PIL Image转换为ComfyUI的tensor格式
    返回格式: (batch, height, width, channels) 值范围0-1
    """
    # 确保是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 转换为numpy数组
    image_np = np.array(image).astype(np.float32) / 255.0

    # 转换为tensor并添加batch维度
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)

    return image_tensor


class JMGeminiImageGenerator:
    """
    ComfyUI custom node for generating images using Google Gemini API
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your Gemini API key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt here"
                }),
                "model": ([GEMINI_3_PRO_MODEL, GEMINI_2_5_FLASH_MODEL], {
                    "default": GEMINI_3_PRO_MODEL
                }),
                "aspect_ratio": ([
                    "1:1", "2:3", "3:2", "3:4", "4:3",
                    "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {
                    "default": "1:1"
                }),
                "resolution": ([
                    "480p", "720p", "768P", "1080p",
                    "1080P", "2K", "4K"
                ], {
                    "default": "2K"
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "JM-Gemini"

    def generate_image(self, gemini_api_key, prompt, model, aspect_ratio, resolution,
                      image1=None, image2=None, image3=None, image4=None, image5=None,
                      image6=None, image7=None, image8=None, image9=None, image10=None):
        """
        主函数：调用Gemini API生成图像
        """
        # 验证API key
        if not gemini_api_key or not gemini_api_key.strip():
            raise ValueError("Gemini API key is required")

        # 收集输入的图像
        input_images = []
        for img in [image1, image2, image3, image4, image5,
                    image6, image7, image8, image9, image10]:
            if img is not None:
                input_images.append(img)

        # 设置输出目录
        output_dir = self._get_output_dir()

        try:
            # 根据是否有输入图像选择生成模式
            if len(input_images) == 0:
                # 文生图模式
                logger.info("[JM-Gemini] Text-to-Image mode")
                generated_image = self._generate_text_to_image(
                    api_key=gemini_api_key,
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    output_dir=output_dir
                )
            else:
                # 图生图/图片编辑模式
                logger.info(f"[JM-Gemini] Image-to-Image mode with {len(input_images)} input images")
                generated_image = self._generate_with_images(
                    api_key=gemini_api_key,
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    input_images=input_images,
                    output_dir=output_dir
                )

            return (generated_image,)

        except Exception as e:
            logger.exception(f"[JM-Gemini] Error generating image: {e}")
            raise RuntimeError(f"Failed to generate image: {str(e)}")

    def _get_output_dir(self):
        """
        获取ComfyUI的output目录
        """
        # ComfyUI的output目录通常在ComfyUI/output
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(os.getcwd(), "output"),
            os.path.join(os.path.dirname(os.getcwd()), "output"),
            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "output"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果都不存在，创建当前目录下的output
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _generate_text_to_image(self, api_key, prompt, model, aspect_ratio,
                               resolution, output_dir):
        """
        文生图模式
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required for text-to-image generation")

        prompt_value = prompt.strip()

        # 创建客户端
        client = genai.Client(api_key=api_key)

        # 根据模型类型配置生成参数
        if model == GEMINI_2_5_FLASH_MODEL:
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio
                )
            )
            resolution_info = ASPECT_RATIO_RESOLUTIONS.get(aspect_ratio, "1024x1024")
        else:
            # Gemini 3 Pro Image
            image_size = RESOLUTION_MAPPING.get(resolution, DEFAULT_RESOLUTION)
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            )
            resolution_info = image_size

        logger.info(f"[JM-Gemini] Calling API with model={model}, aspect_ratio={aspect_ratio}, resolution={resolution_info}")

        # 调用API
        response = client.models.generate_content(
            model=model,
            contents=prompt_value,
            config=config,
        )

        # 处理响应并保存图像
        generated_image = self._process_response(response, model, output_dir, "text2img")

        return generated_image

    def _generate_with_images(self, api_key, prompt, model, aspect_ratio,
                             resolution, input_images, output_dir):
        """
        图生图/图片编辑模式
        """
        # 准备提示词
        if not prompt or not prompt.strip():
            if len(input_images) == 1:
                prompt_value = "Turn this image into a professional quality studio shoot with better lighting and depth of field."
            else:
                prompt_value = "Combine the subjects of these images in a natural way, producing a new image."
        else:
            prompt_value = prompt.strip()

        # 创建客户端
        client = genai.Client(api_key=api_key)

        # 构建contents
        contents = []
        is_single_image = len(input_images) == 1

        if is_single_image:
            # 单图编辑模式：[prompt, image]
            contents.append(prompt_value)
            pil_image = tensor2pil(input_images[0])
            contents.append(pil_image)
            logger.info("[JM-Gemini] Single image edit mode: [prompt, image]")
        else:
            # 多图输入模式：[image1, image2, ..., prompt]
            for img_tensor in input_images:
                pil_image = tensor2pil(img_tensor)
                contents.append(pil_image)
            contents.append(prompt_value)
            logger.info(f"[JM-Gemini] Multi-image mode: [{len(input_images)} images, prompt]")

        # 配置生成参数
        if model == GEMINI_2_5_FLASH_MODEL:
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio
                )
            )
            resolution_info = ASPECT_RATIO_RESOLUTIONS.get(aspect_ratio, "1024x1024")
        else:
            # Gemini 3 Pro Image
            image_size = RESOLUTION_MAPPING.get(resolution, DEFAULT_RESOLUTION)
            config = types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            )
            resolution_info = image_size

        logger.info(f"[JM-Gemini] Calling API with model={model}, aspect_ratio={aspect_ratio}, resolution={resolution_info}")

        # 调用API
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # 处理响应并保存图像
        mode = "imageedit" if is_single_image else "image2image"
        generated_image = self._process_response(response, model, output_dir, mode)

        return generated_image

    def _process_response(self, response, model, output_dir, mode):
        """
        处理Gemini API响应，提取并保存图像
        """
        if not hasattr(response, 'parts') or not response.parts:
            raise RuntimeError("No response parts received from Gemini API")

        logger.info(f"[JM-Gemini] Processing {len(response.parts)} response parts")

        generated_image = None

        for idx, part in enumerate(response.parts):
            # 处理文本响应
            if part.text is not None:
                logger.info(f"[JM-Gemini] Response text: {part.text[:100] if len(part.text) > 100 else part.text}")

            # 尝试提取图像
            try:
                image = part.as_image()
                if image:
                    # 保存图像
                    timestamp = int(time.time())
                    model_prefix = "gemini25flash" if model == GEMINI_2_5_FLASH_MODEL else "gemini3pro"
                    file_name = f"{model_prefix}_{mode}_{timestamp}.png"
                    file_path = os.path.join(output_dir, file_name)

                    image.save(file_path)
                    logger.info(f"[JM-Gemini] Saved image to {file_path}")

                    # 转换为ComfyUI tensor格式
                    generated_image = pil2tensor(image)
                    break  # 只取第一张生成的图像

            except AttributeError:
                # 尝试使用inline_data方式
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    if part.inline_data.data:
                        # 从字节数据创建PIL Image
                        image = Image.open(io.BytesIO(part.inline_data.data))

                        # 保存图像
                        timestamp = int(time.time())
                        model_prefix = "gemini25flash" if model == GEMINI_2_5_FLASH_MODEL else "gemini3pro"
                        file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                        file_name = f"{model_prefix}_{mode}_{timestamp}{file_extension}"
                        file_path = os.path.join(output_dir, file_name)

                        image.save(file_path)
                        logger.info(f"[JM-Gemini] Saved image to {file_path}")

                        # 转换为ComfyUI tensor格式
                        generated_image = pil2tensor(image)
                        break
            except Exception as e:
                logger.warning(f"[JM-Gemini] Could not extract image from part {idx}: {e}")
                continue

        if generated_image is None:
            raise RuntimeError("No images were generated. Please check your prompt and try again.")

        return generated_image


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "JMGeminiImageGenerator": JMGeminiImageGenerator
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiImageGenerator": "JM Gemini Image Generator"
}
