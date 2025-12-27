"""
ComfyUI-JM-Gemini-API Watermark Remover Node
A custom node for ComfyUI that removes watermarks from Gemini generated images
"""

import os
import time
import logging
import numpy as np
from PIL import Image

from .utils import tensor2pil, pil2tensor, get_output_dir

# 设置日志
logger = logging.getLogger(__name__)

# 常量定义
ALPHA_THRESHOLD = 0.002  # 忽略极小的alpha值（噪声）
MAX_ALPHA = 0.99  # 避免除以接近零的值
LOGO_VALUE = 255  # 白色水印的颜色值


def detect_watermark_config(image_width, image_height):
    """
    根据图像尺寸检测水印配置

    Gemini的水印规则：
    - 如果图像宽度和高度都大于1024，使用96×96水印
    - 否则使用48×48水印

    Args:
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        dict: 水印配置 {logo_size, margin_right, margin_bottom}
    """
    if image_width > 1024 and image_height > 1024:
        return {
            "logo_size": 96,
            "margin_right": 64,
            "margin_bottom": 64
        }
    else:
        return {
            "logo_size": 48,
            "margin_right": 32,
            "margin_bottom": 32
        }


def calculate_watermark_position(image_width, image_height, config):
    """
    根据图像尺寸和水印配置计算水印在图像中的位置

    Args:
        image_width: 图像宽度
        image_height: 图像高度
        config: 水印配置

    Returns:
        dict: 水印位置 {x, y, width, height}
    """
    logo_size = config["logo_size"]
    margin_right = config["margin_right"]
    margin_bottom = config["margin_bottom"]

    return {
        "x": image_width - margin_right - logo_size,
        "y": image_height - margin_bottom - logo_size,
        "width": logo_size,
        "height": logo_size
    }


def calculate_alpha_map(bg_image_data):
    """
    从背景捕获图像计算alpha映射

    Args:
        bg_image_data: 背景捕获的numpy数组 (height, width, channels)

    Returns:
        numpy.ndarray: Alpha映射 (值范围0.0-1.0)
    """
    height, width = bg_image_data.shape[:2]
    alpha_map = np.zeros((height, width), dtype=np.float32)

    # 对于每个像素，取RGB三个通道的最大值并归一化到[0, 1]
    for i in range(height):
        for j in range(width):
            r = bg_image_data[i, j, 0]
            g = bg_image_data[i, j, 1]
            b = bg_image_data[i, j, 2]

            # 取RGB三个通道的最大值作为亮度值
            max_channel = max(r, g, b)

            # 归一化到[0, 1]范围
            alpha_map[i, j] = max_channel / 255.0

    return alpha_map


def remove_watermark(image_data, alpha_map, position):
    """
    使用反向alpha混合去除水印

    原理：
    - Gemini添加水印: watermarked = α × logo + (1 - α) × original
    - 反向求解: original = (watermarked - α × logo) / (1 - α)

    Args:
        image_data: 要处理的图像numpy数组 (height, width, channels)
        alpha_map: Alpha通道数据
        position: 水印位置 {x, y, width, height}
    """
    x = position["x"]
    y = position["y"]
    width = position["width"]
    height = position["height"]

    # 处理水印区域的每个像素
    for row in range(height):
        for col in range(width):
            # 计算在原始图像中的索引
            img_row = y + row
            img_col = x + col

            # 获取alpha值
            alpha = alpha_map[row, col]

            # 跳过极小的alpha值（噪声）
            if alpha < ALPHA_THRESHOLD:
                continue

            # 限制alpha值以避免除以接近零的值
            alpha = min(alpha, MAX_ALPHA)
            one_minus_alpha = 1.0 - alpha

            # 对每个RGB通道应用反向alpha混合
            for c in range(3):
                watermarked = image_data[img_row, img_col, c]

                # 反向alpha混合公式
                original = (watermarked - alpha * LOGO_VALUE) / one_minus_alpha

                # 限制在[0, 255]范围内
                image_data[img_row, img_col, c] = np.clip(np.round(original), 0, 255).astype(np.uint8)


class JMGeminiWatermarkRemover:
    """
    ComfyUI自定义节点，用于去除Gemini生成图像上的水印
    """

    def __init__(self):
        # 预加载背景图片
        self.bg_48 = None
        self.bg_96 = None
        self.alpha_map_48 = None
        self.alpha_map_96 = None
        self._load_background_images()

    def _load_background_images(self):
        """加载背景图片并计算alpha映射"""
        try:
            # 获取assets目录路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(os.path.dirname(current_dir), "assets")

            bg_48_path = os.path.join(assets_dir, "bg_48.png")
            bg_96_path = os.path.join(assets_dir, "bg_96.png")

            # 加载48x48背景图
            if os.path.exists(bg_48_path):
                self.bg_48 = Image.open(bg_48_path).convert('RGB')
                bg_48_array = np.array(self.bg_48)
                self.alpha_map_48 = calculate_alpha_map(bg_48_array)
                logger.info("[JM-Gemini] Loaded bg_48.png and calculated alpha map")
            else:
                logger.warning(f"[JM-Gemini] bg_48.png not found at {bg_48_path}")

            # 加载96x96背景图
            if os.path.exists(bg_96_path):
                self.bg_96 = Image.open(bg_96_path).convert('RGB')
                bg_96_array = np.array(self.bg_96)
                self.alpha_map_96 = calculate_alpha_map(bg_96_array)
                logger.info("[JM-Gemini] Loaded bg_96.png and calculated alpha map")
            else:
                logger.warning(f"[JM-Gemini] bg_96.png not found at {bg_96_path}")

        except Exception as e:
            logger.error(f"[JM-Gemini] Error loading background images: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_watermark_from_image"
    CATEGORY = "JM-Gemini"

    def remove_watermark_from_image(self, image):
        """
        主函数：去除图像上的Gemini水印

        Args:
            image: ComfyUI的tensor格式图像

        Returns:
            tuple: 去除水印后的图像tensor
        """
        try:
            # 转换为PIL图像
            pil_image = tensor2pil(image)

            # 获取图像尺寸
            width, height = pil_image.size

            # 检测水印配置
            config = detect_watermark_config(width, height)
            position = calculate_watermark_position(width, height, config)

            logger.info(f"[JM-Gemini] Image size: {width}x{height}, Watermark size: {config['logo_size']}x{config['logo_size']}")
            logger.info(f"[JM-Gemini] Watermark position: ({position['x']}, {position['y']})")

            # 获取对应的alpha映射
            if config["logo_size"] == 48:
                if self.alpha_map_48 is None:
                    raise RuntimeError("bg_48.png not loaded. Cannot remove watermark.")
                alpha_map = self.alpha_map_48
            else:
                if self.alpha_map_96 is None:
                    raise RuntimeError("bg_96.png not loaded. Cannot remove watermark.")
                alpha_map = self.alpha_map_96

            # 转换为numpy数组
            image_array = np.array(pil_image)

            # 去除水印
            remove_watermark(image_array, alpha_map, position)

            # 转换回PIL图像
            processed_image = Image.fromarray(image_array)

            # 保存到output目录
            output_dir = get_output_dir()
            timestamp = int(time.time())
            file_name = f"gemini_watermark_removed_{timestamp}.png"
            file_path = os.path.join(output_dir, file_name)
            processed_image.save(file_path)
            logger.info(f"[JM-Gemini] Saved watermark-removed image to {file_path}")

            # 转换为ComfyUI tensor格式
            output_tensor = pil2tensor(processed_image)

            return (output_tensor,)

        except Exception as e:
            logger.exception(f"[JM-Gemini] Error removing watermark: {e}")
            raise RuntimeError(f"Failed to remove watermark: {str(e)}")


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "JMGeminiWatermarkRemover": JMGeminiWatermarkRemover
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiWatermarkRemover": "JM Gemini Watermark Remover"
}
