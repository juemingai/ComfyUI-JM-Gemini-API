"""
Cookie 配置管理器
用于加载和验证 Gemini Cookie 配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class CookieConfig:
    """Cookie 配置管理"""

    # 默认配置文件路径
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "gemini_cookies.json"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> Dict:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径，默认为 config/gemini_cookies.json

        Returns:
            Dict: 配置字典
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH

        if not path.exists():
            # 创建默认配置
            default_config = {
                "_comment": "Gemini 逆向工程配置文件 - 获取教程: https://github.com/",
                "secure_1psid": "",
                "secure_1psidts": "",
                "snlm0e": "",
                "push_id": "",
                "model_ids": {
                    "flash": "56fdd199312815e2",
                    "pro": "e6fa609c3fa255c0",
                    "thinking": "e051ce1aa80aa576"
                },
                "_获取步骤": {
                    "1": "访问 https://gemini.google.com 并登录",
                    "2": "F12 -> Application -> Cookies",
                    "3": "复制 __Secure-1PSID 和 __Secure-1PSIDTS",
                    "4": "F12 -> Network -> 发送消息",
                    "5": "查找请求头中的 push-id (格式: feeds/xxxxx)",
                    "6": "Ctrl+U 查看源码，搜索 SNlM0e，复制引号内的值"
                }
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, str]:
        """
        验证配置是否完整

        Args:
            config: 配置字典

        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        required_fields = ["secure_1psid", "snlm0e", "push_id"]
        missing = [f for f in required_fields if not config.get(f)]

        if missing:
            error_msg = f"缺少必需字段: {', '.join(missing)}\n\n"
            error_msg += "请按以下步骤获取:\n"
            error_msg += "1. 访问 https://gemini.google.com 并登录\n"
            error_msg += "2. F12 -> Application -> Cookies\n"
            error_msg += "3. 复制 __Secure-1PSID 和 __Secure-1PSIDTS\n"
            error_msg += "4. F12 -> Network -> 发送消息\n"
            error_msg += "5. 查找请求头中的 push-id (格式: feeds/xxxxx)\n"
            error_msg += "6. Ctrl+U 查看源码，搜索 SNlM0e，复制引号内的值\n\n"
            error_msg += f"配置文件路径: {cls.DEFAULT_CONFIG_PATH}"
            return False, error_msg

        # 验证格式
        if config.get("push_id") and not config["push_id"].startswith("feeds/"):
            return False, "push_id 格式错误，应该以 'feeds/' 开头"

        return True, "配置有效"

    @classmethod
    def save(cls, config: Dict, config_path: Optional[Path] = None):
        """
        保存配置文件

        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
