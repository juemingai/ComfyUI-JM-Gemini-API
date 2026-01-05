"""
ComfyUI-JM-Gemini-API Reverse Engineering Node
使用 Gemini 网页版逆向工程方式生成图片
"""

import os
import time
import base64
import logging
from pathlib import Path
from PIL import Image
import io
import re
import json

from .utils import tensor2pil, pil2tensor, get_output_dir
from .gemini_reverse.client import GeminiClient, CookieExpiredError
from .gemini_reverse.config import CookieConfig

# 设置日志
logger = logging.getLogger(__name__)


class JMGeminiReverseGenerator:
    """
    Gemini 逆向工程图片生成节点
    使用浏览器 Cookie 方式调用 Gemini 网页版
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入提示词 (例如: 生成一只可爱的猫咪)"
                }),
                "model": ([
                    "gemini-3.0-flash",
                    "gemini-3.0-pro",
                    "gemini-3.0-flash-thinking"
                ], {
                    "default": "gemini-3.0-flash"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "用于 ComfyUI 重新执行，不传给 Gemini"
                }),
            },
            "optional": {
                "cookies_raw": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "【首次使用】粘贴完整 Cookie，自动保存到配置文件\n格式: __Secure-1PSID=xxx; __Secure-1PSIDTS=xxx; ...\n\n已配置后可留空，系统自动读取配置文件"
                }),
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

    def generate_image(self, prompt, model, seed=0,
                      cookies_raw="",
                      image1=None, image2=None, image3=None, image4=None, image5=None,
                      image6=None, image7=None, image8=None, image9=None, image10=None):
        """
        主生成函数

        Args:
            prompt: 提示词
            model: 模型名称
            seed: 随机种子（仅用于 ComfyUI，不传给 Gemini）
            cookies_raw: 完整的 Cookie 字符串（首次使用时填写）
            image1-image10: 可选的输入图片

        Returns:
            tuple: (IMAGE tensor,)
        """
        logger.info(f"[JM-Gemini-Reverse] 开始生成图片 - 模型: {model}")

        # seed 参数仅用于 ComfyUI 重新执行，不传递给 API

        # 1. 处理 Cookie 输入
        if cookies_raw and cookies_raw.strip():
            logger.info("[JM-Gemini-Reverse] 检测到新的 Cookie，开始自动解析...")
            try:
                # 自动解析 Cookie
                parsed = CookieConfig.parse_cookies_and_fetch_tokens(cookies_raw.strip())

                # 加载现有配置
                config = CookieConfig.load()

                # 更新字段
                config["cookies_raw"] = cookies_raw.strip()
                if parsed.get("secure_1psid"):
                    config["secure_1psid"] = parsed["secure_1psid"]
                    logger.info("[JM-Gemini-Reverse] ✓ 提取 secure_1psid")
                if parsed.get("secure_1psidts"):
                    config["secure_1psidts"] = parsed["secure_1psidts"]
                    logger.info("[JM-Gemini-Reverse] ✓ 提取 secure_1psidts")

                # 处理 SNlM0e：如果自动获取失败，保留配置文件中已有的值
                if parsed.get("snlm0e"):
                    config["snlm0e"] = parsed["snlm0e"]
                    logger.info("[JM-Gemini-Reverse] ✓ 自动获取 snlm0e")
                elif config.get("snlm0e"):
                    logger.info("[JM-Gemini-Reverse] ℹ️  SNlM0e 自动获取失败，保留配置文件中的已有值")
                else:
                    logger.warning("[JM-Gemini-Reverse] ⚠️  SNlM0e 自动获取失败且配置文件中无已有值")
                    logger.warning("[JM-Gemini-Reverse] 请手动从 Network 请求中获取 SNlM0e")
                    logger.warning("[JM-Gemini-Reverse] 详见: config/README_Cookie配置说明.md")

                if parsed.get("push_id"):
                    config["push_id"] = parsed["push_id"]
                    logger.info("[JM-Gemini-Reverse] ✓ 获取 push_id")

                # 保存到配置文件
                CookieConfig.save(config)
                logger.info(f"[JM-Gemini-Reverse] ✓ 配置已保存到 {CookieConfig.DEFAULT_CONFIG_PATH}")

            except Exception as e:
                logger.error(f"[JM-Gemini-Reverse] Cookie 解析失败: {e}")
                raise RuntimeError(
                    f"Cookie 自动解析失败: {e}\n\n"
                    f"请检查:\n"
                    f"1. Cookie 格式是否正确（应包含 __Secure-1PSID）\n"
                    f"2. 网络连接是否正常\n"
                    f"3. Cookie 是否已过期"
                )
        else:
            logger.info("[JM-Gemini-Reverse] 从配置文件加载")

        # 2. 加载配置
        try:
            config = CookieConfig.load()
        except Exception as e:
            raise ValueError(f"加载配置失败: {e}")

        # 3. 验证配置
        valid, msg = CookieConfig.validate(config)
        if not valid:
            raise ValueError(
                f"配置无效: {msg}\n\n"
                f"解决方法:\n"
                f"1. 在节点的 cookies_raw 输入框中粘贴完整 Cookie\n"
                f"2. 或手动编辑配置文件: {CookieConfig.DEFAULT_CONFIG_PATH}"
            )

        # 4. 收集输入图片
        input_images = []
        for img in [image1, image2, image3, image4, image5,
                   image6, image7, image8, image9, image10]:
            if img is not None:
                input_images.append(img)

        if input_images:
            logger.info(f"[JM-Gemini-Reverse] 检测到 {len(input_images)} 张输入图片")

        # 5. 创建客户端
        try:
            client = GeminiClient(
                secure_1psid=config["secure_1psid"],
                secure_1psidts=config.get("secure_1psidts", ""),
                snlm0e=config["snlm0e"],
                push_id=config["push_id"],
                model_ids=config.get("model_ids"),
                debug=False  # 生产环境关闭调试
            )
        except Exception as e:
            raise RuntimeError(f"创建 Gemini 客户端失败: {e}")

        try:
            # 6. 准备图片数据（转换为 base64）
            images_data = []
            if input_images:
                logger.info("[JM-Gemini-Reverse] 转换图片为 base64 格式")
                for i, img_tensor in enumerate(input_images):
                    pil_img = tensor2pil(img_tensor)
                    # 转换为 base64
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="PNG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    images_data.append({
                        "mime_type": "image/png",
                        "data": img_b64
                    })
                    logger.info(f"[JM-Gemini-Reverse] 图片 {i+1}/{len(input_images)} 转换完成")

            # 7. 构建 OpenAI 格式的消息内容
            content = [{"type": "text", "text": prompt}]

            # 添加图片（使用 data URI 格式）
            if images_data:
                logger.info(f"[JM-Gemini-Reverse] 添加 {len(images_data)} 张图片到请求")
                for img_item in images_data:
                    mime_type = img_item["mime_type"]
                    data = img_item["data"]
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"}
                    })

            # 8. 调用 Gemini（使用 OpenAI 格式的 messages 参数）
            logger.info(f"[JM-Gemini-Reverse] 调用 Gemini API - 模型: {model}")
            response = client.chat(
                messages=[{"role": "user", "content": content}],
                model=model,
                reset_context=True  # 每次都是新对话
            )

            # 9. 解析响应文本，提取图片 URL
            reply_text = response.choices[0].message.content
            logger.info(f"[JM-Gemini-Reverse] 收到响应，长度: {len(reply_text)}")

            # 从响应中提取图片（格式: ![alt](/media/gen_xxxxx)）
            media_pattern = r'!\[.*?\]\((/media/[^\)]+)\)'
            media_urls = re.findall(media_pattern, reply_text)

            if not media_urls:
                logger.warning(f"[JM-Gemini-Reverse] 未找到生成的图片，响应内容:\n{reply_text[:500]}")
                raise RuntimeError(
                    f"未在响应中找到生成的图片\n\n"
                    f"响应内容（前500字符）:\n{reply_text[:500]}\n\n"
                    f"可能的原因:\n"
                    f"1. 提示词不符合图片生成要求\n"
                    f"2. Cookie 已过期\n"
                    f"3. 网络问题\n"
                    f"4. Gemini 返回了纯文本回复而非图片"
                )

            logger.info(f"[JM-Gemini-Reverse] 找到 {len(media_urls)} 个媒体文件")

            # 10. 加载第一张生成的图片
            media_path = media_urls[0]  # /media/gen_xxxxx
            media_id = media_path.replace("/media/", "")
            logger.info(f"[JM-Gemini-Reverse] 媒体 ID: {media_id}")

            # 从缓存目录读取
            cache_dir = Path(__file__).parent / "gemini_reverse" / "media_cache"

            # 查找匹配的文件（可能是 png/jpg/gif/webp）
            found_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                file_path = cache_dir / f"{media_id}{ext}"
                if file_path.exists():
                    found_file = file_path
                    logger.info(f"[JM-Gemini-Reverse] 找到缓存文件: {file_path.name}")
                    break

            if not found_file:
                raise RuntimeError(
                    f"无法找到缓存的图片文件: {media_id}\n"
                    f"搜索目录: {cache_dir}\n"
                    f"可能的原因:\n"
                    f"1. 图片下载失败\n"
                    f"2. 缓存目录权限问题\n"
                    f"3. 磁盘空间不足"
                )

            # 11. 读取图片
            logger.info(f"[JM-Gemini-Reverse] 加载图片: {found_file}")
            pil_image = Image.open(found_file)

            # 12. 保存到 ComfyUI output 目录
            output_dir = get_output_dir()
            timestamp = int(time.time())
            model_prefix = model.replace(".", "").replace("-", "")
            output_filename = f"gemini_reverse_{model_prefix}_{timestamp}.png"
            output_path = os.path.join(output_dir, output_filename)
            pil_image.save(output_path)
            logger.info(f"[JM-Gemini-Reverse] 图片已保存: {output_path}")

            # 13. 转换为 ComfyUI tensor
            image_tensor = pil2tensor(pil_image)
            logger.info(f"[JM-Gemini-Reverse] 生成完成，tensor shape: {image_tensor.shape}")

            return (image_tensor,)

        except CookieExpiredError as e:
            error_msg = (
                f"Cookie 已过期或无效:\n{str(e)}\n\n"
                f"解决方法:\n"
                f"1. 访问 https://gemini.google.com 并登录\n"
                f"2. F12 -> Application -> Cookies\n"
                f"3. 复制 __Secure-1PSID, __Secure-1PSIDTS\n"
                f"4. F12 -> Network -> 发送消息 -> 查找 push-id\n"
                f"5. Ctrl+U 查看源码 -> 搜索 SNlM0e\n"
                f"6. 更新配置文件: {CookieConfig.DEFAULT_CONFIG_PATH}"
            )
            logger.error(f"[JM-Gemini-Reverse] {error_msg}")
            raise RuntimeError(error_msg)

        except Exception as e:
            logger.exception(f"[JM-Gemini-Reverse] 生成失败: {e}")
            raise RuntimeError(f"生成失败: {str(e)}")


# 节点注册
NODE_CLASS_MAPPINGS = {
    "JMGeminiReverseGenerator": JMGeminiReverseGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMGeminiReverseGenerator": "JM Gemini Reverse Engineering"
}
