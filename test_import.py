"""
简单的导入测试脚本
用于快速验证 google-genai 是否正常工作
"""

print("测试 google-genai 导入...")

try:
    from google import genai
    print("✓ from google import genai - 成功")
except ImportError as e:
    print(f"✗ from google import genai - 失败: {e}")
    exit(1)

try:
    from google.genai import types
    print("✓ from google.genai import types - 成功")
except ImportError as e:
    print(f"✗ from google.genai import types - 失败: {e}")
    exit(1)

# 检查 ImageConfig
if hasattr(types, 'ImageConfig'):
    print("✓ types.ImageConfig - 存在")

    # 尝试创建一个 ImageConfig 实例
    try:
        config = types.ImageConfig(aspect_ratio="1:1")
        print(f"✓ 创建 ImageConfig 实例 - 成功: {config}")
    except Exception as e:
        print(f"✗ 创建 ImageConfig 实例 - 失败: {e}")
else:
    print("✗ types.ImageConfig - 不存在")
    print("\ntypes 模块中可用的属性:")
    print([x for x in dir(types) if not x.startswith('_')])

# 检查 GenerateContentConfig
if hasattr(types, 'GenerateContentConfig'):
    print("✓ types.GenerateContentConfig - 存在")
else:
    print("✗ types.GenerateContentConfig - 不存在")

print("\n所有测试完成！")
