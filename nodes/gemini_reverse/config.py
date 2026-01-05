"""
Cookie 配置管理器
用于加载和验证 Gemini Cookie 配置
支持粘贴完整 Cookie 字符串自动解析
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import httpx


class CookieConfig:
    """Cookie 配置管理"""

    # 默认配置文件路径
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "gemini_cookies.json"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> Dict:
        """
        加载配置文件，支持自动解析完整 Cookie 字符串

        Args:
            config_path: 配置文件路径，默认为 config/gemini_cookies.json

        Returns:
            Dict: 配置字典
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH

        if not path.exists():
            # 创建默认配置
            default_config = {
                "_comment": "Gemini 逆向工程配置文件",
                "_使用方法": "方法1: 粘贴完整Cookie到cookies_raw字段 | 方法2: 手动填写各字段",

                "cookies_raw": "",
                "_cookies_raw_说明": "【推荐】粘贴浏览器完整Cookie字符串，系统自动解析。粘贴后保存文件，下次加载会自动提取所需字段",

                "secure_1psid": "",
                "secure_1psidts": "",
                "snlm0e": "",
                "push_id": "",

                "model_ids": {
                    "flash": "56fdd199312815e2",
                    "pro": "e6fa609c3fa255c0",
                    "thinking": "e051ce1aa80aa576"
                },

                "_获取步骤_方法1_自动": {
                    "1": "访问 https://gemini.google.com 并登录",
                    "2": "F12 -> Network -> 发送消息 -> 找到任意请求",
                    "3": "复制 Request Headers 中的完整 Cookie 字符串",
                    "4": "粘贴到上面的 cookies_raw 字段",
                    "5": "保存文件，系统会自动解析并填充其他字段"
                },

                "_获取步骤_方法2_手动": {
                    "1": "访问 https://gemini.google.com 并登录",
                    "2": "F12 -> Application -> Cookies",
                    "3": "手动复制 __Secure-1PSID 和 __Secure-1PSIDTS",
                    "4": "F12 -> Network -> 查找 push-id 和 SNlM0e"
                }
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 如果有 cookies_raw，自动解析
        if config.get("cookies_raw") and config["cookies_raw"].strip():
            print("[Cookie配置] 检测到 cookies_raw，开始自动解析...")
            parsed = cls.parse_cookies_and_fetch_tokens(config["cookies_raw"])

            # 更新配置
            if parsed.get("secure_1psid"):
                config["secure_1psid"] = parsed["secure_1psid"]
            if parsed.get("secure_1psidts"):
                config["secure_1psidts"] = parsed["secure_1psidts"]
            if parsed.get("snlm0e"):
                config["snlm0e"] = parsed["snlm0e"]
            if parsed.get("push_id"):
                config["push_id"] = parsed["push_id"]

            # 保存更新后的配置
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"[Cookie配置] 自动解析成功，已更新配置文件")
            except Exception as e:
                print(f"[Cookie配置] 保存配置失败: {e}")

        return config

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

    @classmethod
    def parse_cookies_string(cls, cookies_str: str) -> Dict[str, str]:
        """
        从完整 Cookie 字符串中提取各个字段

        Args:
            cookies_str: 完整的 Cookie 字符串

        Returns:
            Dict: 包含 secure_1psid, secure_1psidts 等字段的字典
        """
        result = {
            "secure_1psid": "",
            "secure_1psidts": "",
        }

        # 解析 Cookie 字符串
        for item in cookies_str.split(";"):
            item = item.strip()
            if "=" in item:
                key, value = item.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key == "__Secure-1PSID":
                    result["secure_1psid"] = value
                elif key == "__Secure-1PSIDTS":
                    result["secure_1psidts"] = value

        return result

    @classmethod
    def fetch_tokens_from_page(cls, cookies_str: str) -> Dict[str, str]:
        """
        从 Gemini 页面自动获取 SNLM0E 和 PUSH_ID

        Args:
            cookies_str: 完整的 Cookie 字符串

        Returns:
            Dict: 包含 snlm0e 和 push_id 的字典
        """
        result = {"snlm0e": "", "push_id": ""}

        try:
            session = httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                }
            )

            # 设置 cookies
            for item in cookies_str.split(";"):
                item = item.strip()
                if "=" in item:
                    key, value = item.split("=", 1)
                    session.cookies.set(key.strip(), value.strip(), domain=".google.com")

            print("[Cookie配置] 正在访问 Gemini 页面...")
            resp = session.get("https://gemini.google.com")
            print(f"[Cookie配置] 响应状态码: {resp.status_code}")

            if resp.status_code != 200:
                print(f"[Cookie配置] 访问失败，状态码: {resp.status_code}")
                # 保存调试页面
                debug_file = Path(__file__).parent / "debug_gemini_page.html"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"[Cookie配置] 响应内容已保存到 {debug_file}")
                return result

            html = resp.text

            # 保存调试页面
            debug_file = Path(__file__).parent / "debug_gemini_page.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[Cookie配置] 页面内容已保存到 {debug_file} (大小: {len(html)} 字节)")

            # 获取 SNLM0E (AT Token)
            snlm0e_patterns = [
                r'"SNlM0e":"([^"]+)"',
                r'SNlM0e["\s:]+["\']([^"\']+)["\']',
                r'"at":"([^"]+)"',
            ]
            for i, pattern in enumerate(snlm0e_patterns):
                match = re.search(pattern, html)
                if match:
                    result["snlm0e"] = match.group(1)
                    print(f"[Cookie配置] 找到 SNLM0E (模式 {i+1}): {result['snlm0e'][:20]}...")
                    break

            if not result["snlm0e"]:
                print("[Cookie配置] 警告: 未找到 SNLM0E Token")
                print("[Cookie配置] 正在搜索可能的位置...")
                # 搜索关键字并打印上下文
                for keyword in ["SNlM0e", "\"at\"", "'at'"]:
                    pos = html.find(keyword)
                    if pos >= 0:
                        print(f"[Cookie配置] 找到关键字 '{keyword}' 在位置 {pos}")
                        context = html[max(0, pos-50):pos+150]
                        print(f"[Cookie配置] 上下文: {context[:200]}")
                        print()

                print("[Cookie配置] ⚠️  建议:")
                print("  1. 检查保存的 debug_gemini_page.html 文件")
                print("  2. 确认 Cookie 是否过期（重新登录 Gemini）")
                print("  3. 手动从页面源码获取 SNlM0e（见配置说明文档）")

            # 获取 PUSH_ID
            push_id_patterns = [
                r'"push[_-]?id["\s:]+["\'](feeds/[a-z0-9]+)["\']',
                r'push[_-]?id["\s:=]+["\'](feeds/[a-z0-9]+)["\']',
                r'feedName["\s:]+["\'](feeds/[a-z0-9]+)["\']',
                r'(feeds/[a-z0-9]{14,})',
            ]
            for pattern in push_id_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    result["push_id"] = matches[0]
                    print(f"[Cookie配置] 找到 PUSH_ID: {result['push_id']}")
                    break

            if not result["push_id"]:
                print("[Cookie配置] 警告: 未找到 PUSH_ID，图片上传功能将不可用")

            return result

        except Exception as e:
            print(f"[Cookie配置] 获取 Token 时出错: {e}")
            return result

    @classmethod
    def parse_cookies_and_fetch_tokens(cls, cookies_str: str) -> Dict[str, str]:
        """
        完整解析流程：提取 Cookie 字段 + 自动获取 Token

        Args:
            cookies_str: 完整的 Cookie 字符串

        Returns:
            Dict: 包含所有字段的字典
        """
        # 步骤1: 从 Cookie 字符串中提取字段
        result = cls.parse_cookies_string(cookies_str)

        # 步骤2: 访问页面自动获取 Token
        tokens = cls.fetch_tokens_from_page(cookies_str)
        result.update(tokens)

        return result
