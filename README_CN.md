# ComfyUI-JM-Gemini-API

一个用于ComfyUI的自定义节点，使用Google Gemini API生成图像，支持文生图和图生图功能。

[English](README.md) | 简体中文

## 功能特性

- 支持多个Gemini模型：
  - `gemini-3-pro-image-preview`（默认，支持2K分辨率）
  - `gemini-2.5-flash-image`
- 文生图（Text-to-Image）
- 图生图（Image-to-Image）支持单张或多张图片输入
- 图片编辑模式（单图输入）
- 可配置的宽高比（1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9）
- 分辨率控制（480p, 720p, 768P, 1080p, 2K, 4K）- 仅适用于gemini-3-pro-image-preview
- 最多支持10张图片输入
- 自动保存图像到ComfyUI的output目录

## 项目结构

```
ComfyUI-JM-Gemini-API/
├── __init__.py              # ComfyUI主入口
├── nodes/                   # 节点实现目录
│   ├── __init__.py         # 节点包初始化
│   ├── jm_gemini_node.py   # Gemini图像生成节点
│   └── README.md           # 节点开发指南
├── requirements.txt         # Python依赖
├── README.md               # 英文文档
├── README_CN.md            # 中文文档（本文件）
└── .gitignore             # Git忽略规则
```

这种模块化结构使得未来添加更多Gemini相关节点变得简单。只需在`nodes/`目录下添加新的节点文件，然后在`nodes/__init__.py`中导入即可。

## 安装说明

### 1. 克隆或下载仓库

将本仓库克隆或下载到ComfyUI的custom_nodes目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-JM-Gemini-API.git
```

### 2. 安装依赖

```bash
cd ComfyUI-JM-Gemini-API
pip install -r requirements.txt
```

### 3. 重启ComfyUI

关闭并重新启动ComfyUI，节点将自动加载。

## 系统要求

- Python 3.8+
- ComfyUI
- google-genai >= 0.2.0
- Pillow >= 10.0.0
- torch
- torchvision
- numpy

## 使用说明

### 获取Gemini API密钥

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 使用您的Google账号登录
3. 创建新的API密钥
4. 复制API密钥，在节点中使用

### 节点参数说明

#### 必需输入参数：

- **gemini_api_key**：您的Gemini API密钥（字符串）
- **prompt**：描述要生成图像的文本提示词（多行文本）
- **model**：选择模型：
  - `gemini-3-pro-image-preview`（默认，支持2K/4K分辨率）
  - `gemini-2.5-flash-image`（速度更快，仅使用宽高比）
- **aspect_ratio**：图像宽高比（1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9）
- **resolution**：图像分辨率（480p, 720p, 768P, 1080p, 2K, 4K）
  - 注意：仅对`gemini-3-pro-image-preview`模型有效
  - 默认值：2K

#### 可选输入参数：

- **image1 ~ image10**：最多10个可选的图像输入，用于图生图
  - 连接Load Image节点的输出
  - 文生图时可以留空

#### 输出：

- **image**：生成的图像（ComfyUI IMAGE tensor格式）
  - 可连接到Preview Image或Save Image节点
  - 自动保存到ComfyUI output目录

### 使用示例

#### 1. 文生图（Text-to-Image）

1. 在工作流中添加"JM Gemini Image Generator"节点
2. 输入您的Gemini API密钥
3. 编写提示词
4. 选择模型和宽高比
5. 保持所有图像输入为空
6. 将输出连接到Preview Image节点
7. 运行工作流

#### 2. 图片编辑（单图输入）

1. 添加Load Image节点并加载图片
2. 添加"JM Gemini Image Generator"节点
3. 将Load Image的输出连接到image1输入
4. 在提示词中输入编辑指令（例如："添加日落背景"）
5. 配置模型和参数
6. 运行工作流

#### 3. 图生图（多图输入）

1. 添加多个Load Image节点
2. 分别连接到image1、image2等输入
3. 输入描述如何组合/转换这些图像的提示词
4. 运行工作流

## 模型区别

### gemini-3-pro-image-preview
- 支持分辨率参数（1K、2K、4K）
- 默认分辨率：2K
- 支持单图编辑
- 输出质量更高

### gemini-2.5-flash-image
- 生成速度更快
- 仅使用宽高比（无分辨率参数）
- 适合快速迭代

## 分辨率映射

对于`gemini-3-pro-image-preview`：
- 480p、720p → 1K
- 768P、1080p、1080P、2K → 2K（默认）
- 4K → 4K

## 常见问题

### 常见错误

1. **"Gemini API key is required"（需要Gemini API密钥）**
   - 确保您已输入有效的API密钥

2. **"No images were generated"（未生成图像）**
   - 检查您的提示词是否清晰且描述详细
   - 尝试不同的宽高比或分辨率
   - 验证API密钥是否有效且有足够的配额

3. **"Failed to generate image"（生成图像失败）**
   - 检查网络连接
   - 验证API密钥权限
   - 检查Gemini API服务状态

4. **图像质量问题**
   - 对于gemini-3-pro-image-preview，尝试使用2K或4K分辨率
   - 让提示词更加详细和具体

## 输出目录

生成的图像自动保存到：
- `ComfyUI/output/`目录
- 文件名格式：`{model}_{mode}_{timestamp}.png`
  - 示例：`gemini3pro_text2img_1234567890.png`

## 添加新节点

如需添加更多Gemini功能节点：

1. 在`nodes/`目录下创建新的节点文件（如`jm_gemini_video_node.py`）
2. 实现节点类并导出映射
3. 在`nodes/__init__.py`中导入并合并映射
4. 重启ComfyUI

详细说明请参考`nodes/README.md`

## 许可证

MIT License

## 鸣谢

作者：JM

基于Google Gemini API开发

## 技术支持

如遇问题或有功能建议，请访问[GitHub仓库](https://github.com/yourusername/ComfyUI-JM-Gemini-API/issues)提交Issue。

## 更新日志

### 版本 1.0.0
- 初始版本发布
- 支持gemini-3-pro-image-preview和gemini-2.5-flash-image模型
- 文生图功能
- 图生图功能（最多10张图片）
- 可配置的宽高比和分辨率
