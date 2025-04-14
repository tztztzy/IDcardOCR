# ID Card OCR

基于PaddleOCR的身份证识别工具

## 安装

1. 确保已安装Python 3.6+
2. 安装依赖:
```
pip install -r requirements.txt
```

## 使用方法

1.创建虚拟环境
 在当前目录下运行
```
python -m venv venv

```
2. 激活虚拟环境
 Windows: .\venv\Scripts\activate
 Linux: source  venv/bin/activate

3. 运行程序:
```
python ocr.py
```
4. 输入身份证图片路径(可直接拖拽图片到窗口)
5. 程序会输出识别结果包括:
   - 姓名
   - 性别
   - 民族
   - 出生日期
   - 住址
   - 身份证号
6. 可视化结果会保存在`tmp/result.jpg`

## 依赖

- PaddleOCR >= 2.6.0.3
- Pillow >= 9.5.0

## 注意事项

- 程序会自动对图片进行增强处理
- 识别结果可能因图片质量而异

## 经过我多次测试，识别准确率基本达到97%以上，只要人眼还可见情况下基本不会出错，该版本为GPU版本，可能需要根据你的cuda版本更换requirements.txt中的链接，例如
```
--extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
或
```
--extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/

```
