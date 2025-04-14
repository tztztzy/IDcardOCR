# ID Card OCR - 身份证智能识别工具

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

基于 PaddleOCR 和 Tauri 开发的高准确率身份证识别工具，支持桌面端应用。

## ✨ 特性

- 🎯 **高准确率**：基于 PaddleOCR v5 模型，文字识别准确
- 🖼️ **图像自动矫正**：集成 C++ 实现的透视变换矫正算法 (感谢开源仓库 [cv_resnet18_card_correction-opencv-dnn](https://github.com/hpc203/cv_resnet18_card_correction-opencv-dnn))
- 🔒 **本地运行**：无需联网，全程本地处理数据，保护隐私
- 🚀 **自动重命名**：自动矫正证件，可以按照需求根据姓名和身份证号自动重命名
- 📊 **批量导出**：支持一键导出含矫正后证件图片的Excel 格式结果

## 🚀 快速开始

### 方式一：使用发行版（推荐）

1. 前往 [Releases](https://github.com/tztztzy/IDcardOCR/releases) 下载对应平台的安装包
2. 安装并运行即可

### 方式二：从源码构建

#### 环境要求

- Node.js 18+
- Rust 1.70+
- Visual Studio 2019+ (Windows) / GCC (Linux/Mac)

#### 1. 克隆仓库

```bash
git clone https://github.com/tztztzy/IDcardOCR.git
cd IDcardOCR
```

#### 2. 构建 C++ 矫正模块 (Windows)

```bash
cd card_correction/cpp
./build_dll.bat
```

#### 3. 构建 Tauri 应用

```bash
cd ../../idcard-ocr/src-tauri

# 安装前端依赖
npm install

（二选一）
# 开发模式运行
npm run tauri dev

# 构建发布版本
npm run tauri build
```

## 📖 使用方法

1. **启动应用**：运行 `ID Card OCR` 桌面应用
2. **加载图片**：
   - 点击"选择图片"按钮
   - 或直接拖拽图片到应用窗口
3. **自动处理**：程序会自动进行图像矫正和 OCR 识别
4. **查看结果**：识别结果会显示在界面上，包括：
   - 姓名
   - 性别
   - 民族
   - 出生日期
   - 住址
   - 身份证号
5. **导出结果**：点击"导出 Excel"按钮保存结果

## 🧠 技术栈

| 模块 | 技术 |
|------|------|
| OCR 引擎 | PaddleOCR v5 (ONNX Runtime) |
| 图像矫正 | OpenCV + C++ |
| 桌面框架 | Tauri v2 |
| 前端 | TypeScript + Vite |
| 后端 | Rust |
| UI | 原生 HTML/CSS |

## 📂 模型文件

项目使用以下 ONNX 模型（位于 `idcard-ocr/src-tauri/models/`）：

- `ch_PP-OCRv5_mobile_det.onnx` - 文本检测模型
- `ch_PP-OCRv5_rec_mobile_infer.onnx` - 文本识别模型
- `ch_ppocr_mobile_v2.0_cls_infer.onnx` - 方向分类模型
- `card_correction.onnx` - 卡片检测模型

## ⚠️ 注意事项

- 识别准确率受图片质量影响，建议使用清晰、光线均匀的图片
- 确保身份证在图片中占主体位置
- 倾斜角度过大的图片可能无法正确矫正
- CPU 版本兼容性较好，GPU 版本依赖较多，暂未完整打包

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
