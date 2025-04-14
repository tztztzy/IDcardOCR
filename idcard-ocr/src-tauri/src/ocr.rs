//! OCR 模块 - 基于 PaddleOCR PP-OCRv5 的文本检测与识别

use std::path::Path;
use std::fs;
use std::cell::RefCell;
use image::DynamicImage;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{inputs, value::Value};
use serde::Serialize;

/// OCR 识别结果
#[derive(Serialize, Clone, Debug)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub box_points: Vec<(f32, f32)>,
}

/// 文本检测框
#[derive(Debug, Clone)]
pub struct TextBox {
    pub points: Vec<(f32, f32)>,
    pub score: f32,
}

/// OCR 引擎
pub struct OcrEngine {
    det_session: RefCell<Session>,
    rec_session: RefCell<Session>,
    char_dict: Vec<String>,
}

impl OcrEngine {
    /// 创建 OCR 引擎
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let det_model_path = Path::new(models_dir).join("ch_PP-OCRv5_mobile_det.onnx");
        let rec_model_path = Path::new(models_dir).join("ch_PP-OCRv5_rec_mobile_infer.onnx");
        let dict_path = Path::new(models_dir).join("ppocrv5_dict.txt");

        // 检查文件是否存在
        if !det_model_path.exists() {
            return Err(format!("检测模型不存在: {:?}", det_model_path));
        }
        if !rec_model_path.exists() {
            return Err(format!("识别模型不存在: {:?}", rec_model_path));
        }
        if !dict_path.exists() {
            return Err(format!("字典文件不存在: {:?}", dict_path));
        }

        // 加载 ONNX 模型
        let det_session = Session::builder()
            .map_err(|e| format!("创建检测会话失败: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("设置优化级别失败: {}", e))?
            .commit_from_file(&det_model_path)
            .map_err(|e| format!("加载检测模型失败: {}", e))?;

        let rec_session = Session::builder()
            .map_err(|e| format!("创建识别会话失败: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("设置优化级别失败: {}", e))?
            .commit_from_file(&rec_model_path)
            .map_err(|e| format!("加载识别模型失败: {}", e))?;

        // 加载字典 - PaddleOCR PP-OCRv5 字典格式
        // 字典第1行对应索引1，索引0是 CTC blank（不输出任何字符）
        // 所以字典大小应该是 num_classes - 1
        let dict_content = fs::read_to_string(&dict_path)
            .map_err(|e| format!("读取字典文件失败: {}", e))?;
        
        // 索引0是 blank（CTC blank token，不输出字符）
        let mut char_dict = vec!["".to_string()]; // 索引0 = blank，不输出
        
        for line in dict_content.lines() {
            // 每行一个字符，直接添加
            char_dict.push(line.to_string());
        }

        eprintln!("OCR引擎初始化成功，字典大小: {}, 模型类别数应为: {}", char_dict.len(), char_dict.len());

        Ok(OcrEngine {
            det_session: RefCell::new(det_session),
            rec_session: RefCell::new(rec_session),
            char_dict,
        })
    }

    /// 处理图像，返回识别结果
    pub fn process_image(&self, image: &DynamicImage) -> Result<Vec<OcrResult>, String> {
        let (width, height) = (image.width(), image.height());

        // 1. 文本检测
        let text_boxes = self.detect_text(image)?;

        if text_boxes.is_empty() {
            eprintln!("未检测到文本区域");
            return Ok(vec![]);
        }

        eprintln!("检测到 {} 个文本区域", text_boxes.len());

        // 2. 按位置排序（从上到下，从左到右）
        let mut sorted_boxes = text_boxes.clone();
        sorted_boxes.sort_by(|a, b| {
            let a_y = a.points.iter().map(|p| p.1).sum::<f32>() / a.points.len() as f32;
            let b_y = b.points.iter().map(|p| p.1).sum::<f32>() / b.points.len() as f32;
            let a_x = a.points.iter().map(|p| p.0).sum::<f32>() / a.points.len() as f32;
            let b_x = b.points.iter().map(|p| p.0).sum::<f32>() / b.points.len() as f32;
            // 先按 y 排序，如果 y 差距小于 10，则按 x 排序
            if (a_y - b_y).abs() < 10.0 {
                a_x.partial_cmp(&b_x).unwrap()
            } else {
                a_y.partial_cmp(&b_y).unwrap()
            }
        });

        // 3. 对每个文本框进行识别
        let mut results = Vec::new();
        for (i, box_item) in sorted_boxes.iter().enumerate() {
            // 裁剪文本区域
            let cropped = self.crop_text_region(image, &box_item.points, width, height)?;
            
            // 识别文本
            let (text, confidence) = self.recognize_text(&cropped)?;

            eprintln!("文本区域 {}: '{}' (置信度: {:.2})", i, text, confidence);

            if !text.is_empty() {
                results.push(OcrResult {
                    text,
                    confidence,
                    box_points: box_item.points.clone(),
                });
            }
        }

        Ok(results)
    }

    /// 文本检测
    fn detect_text(&self, image: &DynamicImage) -> Result<Vec<TextBox>, String> {
        // 预处理
        let (shape, data, ratio_h, ratio_w) = self.preprocess_det(image)?;

        // 创建输入值
        let input_value = Value::from_array((shape, data))
            .map_err(|e| format!("创建检测输入失败: {}", e))?;

        // 推理
        let mut session = self.det_session.borrow_mut();
        let output_name = session.outputs()[0].name().to_string();
        let outputs = session
            .run(inputs![input_value])
            .map_err(|e| format!("检测推理失败: {}", e))?;

        // 获取输出
        let output = outputs.get(output_name.as_str())
            .ok_or("检测输出为空")?;

        let (out_shape, out_data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("提取检测输出失败: {}", e))?;

        // 后处理
        let boxes = self.postprocess_det(out_shape, out_data, ratio_h, ratio_w)?;

        Ok(boxes)
    }

    /// 检测模型预处理
    fn preprocess_det(&self, image: &DynamicImage) -> Result<(Vec<usize>, Vec<f32>, f32, f32), String> {
        let (ori_h, ori_w) = (image.height() as f32, image.width() as f32);
        
        // 计算缩放比例，保持长宽比
        let max_size = 960.0;
        let scale = max_size / ori_h.max(ori_w);
        let scale = scale.min(1.0);
        
        let new_h = (ori_h * scale).round() as u32;
        let new_w = (ori_w * scale).round() as u32;
        
        // 确保尺寸是32的倍数
        let new_h = ((new_h + 31) / 32) * 32;
        let new_w = ((new_w + 31) / 32) * 32;

        let ratio_h = ori_h / new_h as f32;
        let ratio_w = ori_w / new_w as f32;

        // 缩放图像
        let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb_image = resized.to_rgb8();

        // 转换为 tensor，归一化 - CHW 格式
        // PaddleOCR 检测模型使用的归一化参数
        let mut input_data = Vec::with_capacity((new_h * new_w * 3) as usize);
        
        // 按通道存储 (CHW 格式)
        for c in 0..3 {
            for y in 0..new_h {
                for x in 0..new_w {
                    let pixel = rgb_image.get_pixel(x, y);
                    // PaddleOCR 检测模型归一化参数: (x / 255 - 0.485) / 0.229 等
                    let mean = [0.485, 0.456, 0.406][c];
                    let std = [0.229, 0.224, 0.225][c];
                    let normalized = (pixel[c] as f32 / 255.0 - mean) / std;
                    input_data.push(normalized);
                }
            }
        }

        let shape = vec![1, 3, new_h as usize, new_w as usize];

        Ok((shape, input_data, ratio_h, ratio_w))
    }

    /// 检测后处理 - DBNet
    fn postprocess_det(
        &self,
        shape: &ort::tensor::Shape,
        data: &[f32],
        ratio_h: f32,
        ratio_w: f32,
    ) -> Result<Vec<TextBox>, String> {
        // 输出形状通常是 [1, 1, H, W]
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        
        if dims.len() < 3 {
            return Err(format!("检测输出维度不正确: {:?}", dims));
        }

        let h = dims[dims.len() - 2];
        let w = dims[dims.len() - 1];
        
        // 提取二值图
        let mut bitmap = vec![false; h * w];
        let threshold = 0.3;
        
        for i in 0..h * w {
            let val = data[i];
            bitmap[i] = val > threshold;
        }

        // 使用连通域检测找到文本框
        let boxes = self.find_text_boxes(&bitmap, w, h, ratio_w, ratio_h);

        Ok(boxes)
    }

    /// 查找文本框
    fn find_text_boxes(
        &self,
        bitmap: &[bool],
        width: usize,
        height: usize,
        ratio_w: f32,
        ratio_h: f32,
    ) -> Vec<TextBox> {
        let mut boxes = Vec::new();
        let mut visited = vec![false; width * height];
        
        // 连通域检测
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if bitmap[idx] && !visited[idx] {
                    // BFS 查找连通域
                    let mut region = Vec::new();
                    let mut queue = vec![(x, y)];
                    
                    while let Some((cx, cy)) = queue.pop() {
                        let cidx = cy * width + cx;
                        if visited[cidx] || !bitmap[cidx] {
                            continue;
                        }
                        visited[cidx] = true;
                        region.push((cx, cy));
                        
                        // 8邻域
                        for dx in -1i32..=1 {
                            for dy in -1i32..=1 {
                                if dx == 0 && dy == 0 { continue; }
                                let nx = cx as i32 + dx;
                                let ny = cy as i32 + dy;
                                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                    queue.push((nx as usize, ny as usize));
                                }
                            }
                        }
                    }
                    
                    // 过滤小区域
                    if region.len() < 50 {
                        continue;
                    }
                    
                    // 计算边界框
                    let min_x = region.iter().map(|(x, _)| *x).min().unwrap() as f32;
                    let max_x = region.iter().map(|(x, _)| *x).max().unwrap() as f32;
                    let min_y = region.iter().map(|(_, y)| *y).min().unwrap() as f32;
                    let max_y = region.iter().map(|(_, y)| *y).max().unwrap() as f32;
                    
                    // 扩展边界
                    let padding = 3.0;
                    let min_x = (min_x - padding).max(0.0);
                    let max_x = (max_x + padding).min(width as f32 - 1.0);
                    let min_y = (min_y - padding).max(0.0);
                    let max_y = (max_y + padding).min(height as f32 - 1.0);
                    
                    // 转换回原始坐标
                    let points = vec![
                        (min_x * ratio_w, min_y * ratio_h),
                        (max_x * ratio_w, min_y * ratio_h),
                        (max_x * ratio_w, max_y * ratio_h),
                        (min_x * ratio_w, max_y * ratio_h),
                    ];
                    
                    boxes.push(TextBox {
                        points,
                        score: 0.9,
                    });
                }
            }
        }

        boxes
    }

    /// 裁剪文本区域
    fn crop_text_region(
        &self,
        image: &DynamicImage,
        points: &[(f32, f32)],
        _max_w: u32,
        _max_h: u32,
    ) -> Result<DynamicImage, String> {
        // 计算边界框
        let min_x = points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min).max(0.0) as u32;
        let min_y = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min).max(0.0) as u32;
        let max_x = points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max).min(image.width() as f32) as u32;
        let max_y = points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max).min(image.height() as f32) as u32;

        if max_x <= min_x || max_y <= min_y {
            return Err("无效的裁剪区域".to_string());
        }

        let cropped = image.clone().crop(min_x, min_y, max_x - min_x, max_y - min_y);
        Ok(cropped)
    }

    /// 文本识别
    fn recognize_text(&self, image: &DynamicImage) -> Result<(String, f32), String> {
        // 预处理
        let (shape, data) = self.preprocess_rec(image)?;

        // 创建输入值
        let input_value = Value::from_array((shape, data))
            .map_err(|e| format!("创建识别输入失败: {}", e))?;

        // 推理
        let mut session = self.rec_session.borrow_mut();
        let output_name = session.outputs()[0].name().to_string();
        let outputs = session
            .run(inputs![input_value])
            .map_err(|e| format!("识别推理失败: {}", e))?;

        // 获取输出
        let output = outputs.get(output_name.as_str())
            .ok_or("识别输出为空")?;

        let (out_shape, out_data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("提取识别输出失败: {}", e))?;

        // CTC 解码
        let (text, confidence) = self.ctc_decode(out_shape, out_data);

        Ok((text, confidence))
    }

    /// 识别模型预处理
    fn preprocess_rec(&self, image: &DynamicImage) -> Result<(Vec<usize>, Vec<f32>), String> {
        // 识别模型输入高度固定为48，宽度按比例缩放，最大320
        let target_h = 48;
        let max_w = 320;
        
        let (ori_h, ori_w) = (image.height(), image.width());
        let ratio = ori_w as f32 / ori_h as f32;
        let new_w = ((target_h as f32 * ratio).min(max_w as f32) as u32).max(10);
        
        // 确保宽度是4的倍数
        let new_w = ((new_w + 3) / 4) * 4;

        // 缩放图像
        let resized = image.resize_exact(new_w, target_h, image::imageops::FilterType::Triangle);
        let rgb_image = resized.to_rgb8();

        // 转换为 tensor - CHW 格式
        // PaddleOCR 识别模型归一化参数: (x / 255 - 0.5) / 0.5
        let mut input_data = Vec::with_capacity((target_h * new_w * 3) as usize);
        
        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..new_w {
                    let pixel = rgb_image.get_pixel(x, y);
                    let normalized = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
                    input_data.push(normalized);
                }
            }
        }

        let shape = vec![1, 3, target_h as usize, new_w as usize];

        Ok((shape, input_data))
    }

    /// CTC 解码
    fn ctc_decode(&self, shape: &ort::tensor::Shape, data: &[f32]) -> (String, f32) {
        // 输出形状通常是 [1, seq_len, num_classes] 或 [seq_len, num_classes]
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        
        if dims.is_empty() {
            return (String::new(), 0.0);
        }

        let (seq_len, num_classes) = if dims.len() == 2 {
            (dims[0], dims[1])
        } else if dims.len() >= 3 {
            (dims[dims.len() - 2], dims[dims.len() - 1])
        } else {
            return (String::new(), 0.0);
        };

        eprintln!("CTC解码: seq_len={}, num_classes={}, dict_len={}", seq_len, num_classes, self.char_dict.len());

        let mut text = String::new();
        let mut total_conf = 0.0;
        let mut count = 0;
        let mut last_idx: Option<usize> = None; // 上一个非 blank 的索引

        for t in 0..seq_len {
            // 找到当前时间步的最大概率字符
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;

            for c in 0..num_classes {
                let val = data[t * num_classes + c];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }

            // CTC 解码规则:
            // 1. blank (索引0) 不输出
            // 2. 连续相同的字符合并为一个
            // 3. blank 允许输出重复字符
            if max_idx != 0 {
                // 不是 blank
                if last_idx != Some(max_idx) {
                    // 与上一个字符不同，可以输出
                    if max_idx < self.char_dict.len() {
                        let ch = &self.char_dict[max_idx];
                        text.push_str(ch);
                        total_conf += max_val.exp();
                        count += 1;
                    }
                }
                last_idx = Some(max_idx);
            } else {
                // 遇到 blank，重置上一个字符
                last_idx = None;
            }
        }

        let avg_conf = if count > 0 {
            total_conf / count as f32
        } else {
            0.0
        };

        (text, avg_conf)
    }

    /// 处理 base64 编码的图像
    pub fn process_base64(&self, image_base64: &str) -> Result<Vec<OcrResult>, String> {
        use base64::{engine::general_purpose, Engine};

        // 解码 base64
        let image_data = general_purpose::STANDARD
            .decode(image_base64)
            .map_err(|e| format!("Base64 解码失败: {}", e))?;

        // 加载图像
        let img = image::load_from_memory(&image_data)
            .map_err(|e| format!("图像加载失败: {}", e))?;

        self.process_image(&img)
    }
}
