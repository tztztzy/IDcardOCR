// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use base64::{engine::general_purpose, Engine};
use serde::{Serialize, Deserialize};
use tauri::State;
use std::io::{BufWriter, Cursor, Write};
use std::collections::HashSet;
use chrono;

// printpdf types
use printpdf::{Mm, Px, PdfDocument, PdfLayerReference, Image, ImageXObject, ColorSpace, ColorBits, ImageTransform, BuiltinFont};
use rust_xlsxwriter::{Workbook, Format, FormatAlign};

mod correction_dll;
use correction_dll::{CardCorrectionDll, CorrectionResult as DllCorrectionResult};

mod ocr;
use ocr::OcrEngine;

// ========== 数据结构定义 ==========

/// 身份证字段
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IdCardFields {
    pub name: Option<String>,
    pub gender: Option<String>,
    pub ethnicity: Option<String>,
    pub birth_date: Option<String>,
    pub address: Option<String>,
    pub id_number: Option<String>,
}

/// 单张身份证识别结果
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RecognitionResult {
    pub filename: String,
    pub original_image: String,
    pub corrected_image: String,
    pub texts: Vec<String>,
    pub confidence: f32,
    pub fields: IdCardFields,
}

/// 批量身份证识别结果
#[derive(Serialize, Clone, Debug)]
pub struct BatchRecognitionResult {
    pub results: Vec<RecognitionResult>,
    pub total: usize,
    pub success: usize,
    pub failed: usize,
}

/// 矫正结果
#[derive(Serialize, Clone, Debug)]
pub struct CorrectionResult {
    pub filename: String,
    pub original_image: String,
    pub corrected_image: String,
}

/// OCR文本结果
#[derive(Serialize, Clone, Debug)]
pub struct OcrTextResult {
    pub filename: String,
    pub texts: Vec<String>,
    pub confidence: f32,
}

/// OCR服务状态
#[derive(Serialize, Debug)]
pub struct OcrServiceStatus {
    pub ready: bool,
    pub message: String,
    pub ocr_engine_ready: bool,
}

/// 文本行结果
#[derive(Debug, Clone)]
struct TextLineResult {
    text: String,
    score: f32,
}

// ========== 矫正结果类型 ==========

#[derive(Serialize, Clone, Debug)]
pub struct UnifiedCorrectionResult {
    pub polygons: Vec<Vec<f32>>,
    pub bbox: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
    pub output_images: Vec<String>,
    pub labels: Vec<i32>,
    pub layout: Vec<i32>,
    pub center: Vec<Vec<f32>>,
}

impl From<DllCorrectionResult> for UnifiedCorrectionResult {
    fn from(result: DllCorrectionResult) -> Self {
        let polygons: Vec<Vec<f32>> = result.detections.iter().map(|d| {
            vec![
                d.polygon[0][0], d.polygon[0][1],
                d.polygon[1][0], d.polygon[1][1],
                d.polygon[2][0], d.polygon[2][1],
                d.polygon[3][0], d.polygon[3][1],
            ]
        }).collect();

        let bbox: Vec<Vec<f32>> = result.detections.iter().map(|d| {
            let x_coords: Vec<f32> = d.polygon.iter().map(|p| p[0]).collect();
            let y_coords: Vec<f32> = d.polygon.iter().map(|p| p[1]).collect();
            vec![
                x_coords.iter().cloned().fold(f32::INFINITY, f32::min),
                y_coords.iter().cloned().fold(f32::INFINITY, f32::min),
                x_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                y_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ]
        }).collect();

        let scores: Vec<f32> = result.detections.iter().map(|d| d.score).collect();
        let labels: Vec<i32> = result.detections.iter().map(|d| d.angle).collect();
        let layout: Vec<i32> = result.detections.iter().map(|d| d.ftype).collect();
        let center: Vec<Vec<f32>> = result.detections.iter().map(|d| {
            vec![d.center[0], d.center[1]]
        }).collect();

        let output_images: Vec<String> = result.output_images.iter().map(|img| {
            general_purpose::STANDARD.encode(&img.data)
        }).collect();

        UnifiedCorrectionResult {
            polygons,
            bbox,
            scores,
            output_images,
            labels,
            layout,
            center,
        }
    }
}

/// DLL 矫正器
pub struct DllCorrector {
    dll: CardCorrectionDll,
}

impl DllCorrector {
    pub fn new(dll_path: &str, model_path: &str) -> Result<Self, String> {
        let dll = CardCorrectionDll::new(dll_path)?;
        let _model = correction_dll::CardCorrectionModel::new(&dll, model_path)?;
        Ok(DllCorrector { dll })
    }

    pub fn process_image(&self, image_base64: &str) -> Result<UnifiedCorrectionResult, String> {
        let model_path = get_model_path();
        let model = correction_dll::CardCorrectionModel::new(&self.dll, &model_path)?;
        model.process_base64(image_base64)
            .map(|r| r.into())
    }
}

/// 应用状态
struct AppState {
    corrector: Mutex<Option<DllCorrector>>,
    ocr_engine: Mutex<Option<OcrEngine>>,
}

// ========== Tauri Commands ==========

#[tauri::command]
fn check_ocr_status(state: State<AppState>) -> Result<OcrServiceStatus, String> {
    let corrector = state.corrector.lock().map_err(|e| e.to_string())?;
    let ocr_engine = state.ocr_engine.lock().map_err(|e| e.to_string())?;

    let corrector_ready = corrector.is_some();
    let ocr_engine_ready = ocr_engine.is_some();

    if corrector_ready && ocr_engine_ready {
        Ok(OcrServiceStatus {
            ready: true,
            message: "服务就绪".to_string(),
            ocr_engine_ready,
        })
    } else {
        let mut missing = Vec::new();
        if !corrector_ready { missing.push("矫正器"); }
        if !ocr_engine_ready { missing.push("OCR引擎"); }
        Ok(OcrServiceStatus {
            ready: false,
            message: format!("{}未就绪", missing.join("、")),
            ocr_engine_ready,
        })
    }
}

/// 证件矫正 - 只矫正不识别
#[tauri::command]
fn correct_image(
    state: State<AppState>,
    image_base64: String,
    filename: String,
) -> Result<CorrectionResult, String> {
    let corrector = state.corrector.lock().map_err(|e| e.to_string())?;
    
    if let Some(ref c) = *corrector {
        let result = c.process_image(&image_base64)?;
        if !result.output_images.is_empty() {
            Ok(CorrectionResult {
                filename,
                original_image: image_base64,
                corrected_image: result.output_images[0].clone(),
            })
        } else {
            // 未检测到证件，返回原图
            Ok(CorrectionResult {
                filename,
                original_image: image_base64.clone(),
                corrected_image: image_base64,
            })
        }
    } else {
        Err("矫正器未初始化".to_string())
    }
}

/// 身份证识别 - 矫正 + OCR + 字段提取
#[tauri::command]
fn recognize_idcard(
    state: State<AppState>,
    image_base64: String,
    filename: String,
) -> Result<RecognitionResult, String> {
    // 第一步：矫正图片
    let corrected_image_base64 = {
        let corrector = state.corrector.lock().map_err(|e| e.to_string())?;
        if let Some(ref c) = *corrector {
            match c.process_image(&image_base64) {
                Ok(result) if !result.output_images.is_empty() => result.output_images[0].clone(),
                _ => image_base64.clone(),
            }
        } else {
            image_base64.clone()
        }
    };

    // 第二步：使用内置 OCR 引擎识别
    let ocr_engine = state.ocr_engine.lock().map_err(|e| e.to_string())?;
    
    let engine = ocr_engine.as_ref()
        .ok_or_else(|| "OCR引擎未初始化".to_string())?;
    
    let ocr_results = engine.process_base64(&corrected_image_base64)
        .map_err(|e| format!("OCR识别失败: {}", e))?;
    
    // 转换为 TextLineResult 格式
    let text_results: Vec<TextLineResult> = ocr_results.iter().map(|r| TextLineResult {
        text: r.text.clone(),
        score: r.confidence,
    }).collect();

    // 第三步：解析身份证字段
    let fields = extract_idcard_fields(&text_results);

    // 计算平均置信度
    let confidence = if text_results.is_empty() {
        0.0
    } else {
        text_results.iter()
            .map(|t| t.score)
            .sum::<f32>() / text_results.len() as f32
    };

    Ok(RecognitionResult {
        filename,
        original_image: image_base64,
        corrected_image: corrected_image_base64,
        texts: text_results.into_iter().map(|t| t.text).collect(),
        confidence,
        fields,
    })
}

/// 通用OCR - 只进行文字识别
#[tauri::command]
fn ocr_image(
    state: State<AppState>,
    image_base64: String,
    filename: String,
) -> Result<OcrTextResult, String> {
    let ocr_engine = state.ocr_engine.lock().map_err(|e| e.to_string())?;
    
    let engine = ocr_engine.as_ref()
        .ok_or_else(|| "OCR引擎未初始化".to_string())?;
    
    let ocr_results = engine.process_base64(&image_base64)
        .map_err(|e| format!("OCR识别失败: {}", e))?;
    
    let texts: Vec<String> = ocr_results.iter().map(|r| r.text.clone()).collect();
    
    let confidence = if ocr_results.is_empty() {
        0.0
    } else {
        ocr_results.iter()
            .map(|r| r.confidence)
            .sum::<f32>() / ocr_results.len() as f32
    };

    Ok(OcrTextResult {
        filename,
        texts,
        confidence,
    })
}

/// 批量身份证识别 - 矫正 + OCR + 字段提取
/// 输入: 图片列表，每个元素包含 filename 和 image_base64
/// 输出: BatchRecognitionResult
#[tauri::command]
fn batch_recognize_idcard(
    state: State<AppState>,
    images: Vec<ImageInput>,
) -> Result<BatchRecognitionResult, String> {
    let total = images.len();
    let mut results = Vec::with_capacity(total);
    let mut success = 0;
    let mut failed = 0;

    for image_input in images {
        match process_single_idcard(&state, &image_input.image_base64, &image_input.filename) {
            Ok(result) => {
                results.push(result);
                success += 1;
            }
            Err(e) => {
                eprintln!("处理 {} 失败: {}", image_input.filename, e);
                failed += 1;
                // 添加一个空结果表示失败
                let original = image_input.image_base64.clone();
                results.push(RecognitionResult {
                    filename: image_input.filename,
                    original_image: original.clone(),
                    corrected_image: original,
                    texts: vec![format!("处理失败: {}", e)],
                    confidence: 0.0,
                    fields: IdCardFields::default(),
                });
            }
        }
    }

    Ok(BatchRecognitionResult {
        results,
        total,
        success,
        failed,
    })
}

/// 图片输入结构
#[derive(Serialize, Clone, Debug, serde::Deserialize)]
pub struct ImageInput {
    pub filename: String,
    pub image_base64: String,
}

/// Excel 导出结果
#[derive(Serialize, Clone, Debug)]
pub struct ExcelExportResult {
    pub success: bool,
    pub message: String,
    pub excel_base64: Option<String>,
    pub total: usize,
    pub written: usize,
    pub skipped: usize,
}

/// PDF 生成结果
#[derive(Serialize, Clone, Debug)]
pub struct PdfGenerateResult {
    pub success: bool,
    pub message: String,
    pub pdf_base64: Option<String>,
}

/// ZIP 下载选项
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZipDownloadOptions {
    pub auto_rename: bool,
    pub include_name: bool,
    pub include_id_number: bool,
}

/// ZIP 下载结果
#[derive(Serialize, Clone, Debug)]
pub struct ZipDownloadResult {
    pub success: bool,
    pub message: String,
    pub zip_base64: Option<String>,
    pub filename: String,
}

/// 图片项（用于ZIP下载）
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ImageItem {
    pub filename: String,
    pub image_base64: String,
    pub fields: Option<IdCardFields>,
}

/// 处理单张身份证图片
fn process_single_idcard(
    state: &AppState,
    image_base64: &str,
    filename: &str,
) -> Result<RecognitionResult, String> {
    // 第一步：矫正图片
    let corrected_image_base64 = {
        let corrector = state.corrector.lock().map_err(|e| e.to_string())?;
        if let Some(ref c) = *corrector {
            match c.process_image(image_base64) {
                Ok(result) if !result.output_images.is_empty() => result.output_images[0].clone(),
                _ => image_base64.to_string(),
            }
        } else {
            image_base64.to_string()
        }
    };

    // 第二步：使用内置 OCR 引擎识别
    let ocr_engine = state.ocr_engine.lock().map_err(|e| e.to_string())?;
    
    let engine = ocr_engine.as_ref()
        .ok_or_else(|| "OCR引擎未初始化".to_string())?;
    
    let ocr_results = engine.process_base64(&corrected_image_base64)
        .map_err(|e| format!("OCR识别失败: {}", e))?;
    
    // 转换为 TextLineResult 格式
    let text_results: Vec<TextLineResult> = ocr_results.iter().map(|r| TextLineResult {
        text: r.text.clone(),
        score: r.confidence,
    }).collect();

    // 第三步：解析身份证字段
    let fields = extract_idcard_fields(&text_results);

    // 计算平均置信度
    let confidence = if text_results.is_empty() {
        0.0
    } else {
        text_results.iter()
            .map(|t| t.score)
            .sum::<f32>() / text_results.len() as f32
    };

    Ok(RecognitionResult {
        filename: filename.to_string(),
        original_image: image_base64.to_string(),
        corrected_image: corrected_image_base64,
        texts: text_results.into_iter().map(|t| t.text).collect(),
        confidence,
        fields,
    })
}

/// 从OCR结果中提取身份证字段
fn extract_idcard_fields(results: &[TextLineResult]) -> IdCardFields {
    // 将所有文本合并为一个字符串，方便正则匹配
    let all_text: String = results.iter()
        .map(|t| t.text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<&str>>()
        .join("");

    let mut fields = IdCardFields::default();

    // 使用正则表达式提取各字段
    // 姓名提取: 姓名(.+?)(:?性|别) 等
    let name_patterns = [
        r"姓名(.+?)(?:性|别)",
        r"名(.+?)(?:性|别)",
        r"姓(.+?)(?:性|别)",
    ];
    for pattern in &name_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(&all_text) {
                if let Some(m) = caps.get(1) {
                    let name = m.as_str().trim();
                    if !name.is_empty() && name.len() <= 10 {
                        fields.name = Some(name.to_string());
                        break;
                    }
                }
            }
        }
    }

    // 性别提取: 性别([男女]) 等
    let gender_patterns = [
        r"性别([男女])",
        r"别([男女])",
        r"性([男女])",
    ];
    for pattern in &gender_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(&all_text) {
                if let Some(m) = caps.get(1) {
                    fields.gender = Some(m.as_str().to_string());
                    break;
                }
            }
        }
    }

    // 民族提取: 民族(.+?)(:?出|生) 等
    let ethnicity_patterns = [
        r"民族(.+?)(?:出|生)",
        r"族(.+?)(?:出|生)",
        r"民(.+?)(?:出|生)",
    ];
    for pattern in &ethnicity_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(&all_text) {
                if let Some(m) = caps.get(1) {
                    let ethnicity = m.as_str().trim();
                    if !ethnicity.is_empty() {
                        // 确保民族以"族"结尾
                        if ethnicity.ends_with("族") {
                            fields.ethnicity = Some(ethnicity.to_string());
                        } else {
                            fields.ethnicity = Some(format!("{}族", ethnicity));
                        }
                        break;
                    }
                }
            }
        }
    }

    // 出生日期提取: 出生(.+?)(:?住|址) 等
    let birth_patterns = [
        r"出生(.+?)(?:住|址)",
        r"生(.+?)(?:住|址)",
        r"出(.+?)(?:住|址)",
    ];
    for pattern in &birth_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(&all_text) {
                if let Some(m) = caps.get(1) {
                    let birth_str = m.as_str().trim();
                    // 尝试解析日期格式
                    if let Some(formatted_date) = parse_birth_date(birth_str) {
                        fields.birth_date = Some(formatted_date);
                        break;
                    }
                }
            }
        }
    }

    // 如果上面没匹配到，尝试直接匹配日期格式
    if fields.birth_date.is_none() {
        if let Ok(re) = regex::Regex::new(r"(\d{4})[年\-\.](\d{1,2})[月\-\.](\d{1,2})日?") {
            if let Some(caps) = re.captures(&all_text) {
                let year = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let month = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                let day = caps.get(3).map(|m| m.as_str()).unwrap_or("");
                let month_padded = if month.len() == 1 { format!("0{}", month) } else { month.to_string() };
                let day_padded = if day.len() == 1 { format!("0{}", day) } else { day.to_string() };
                fields.birth_date = Some(format!("{}年{}月{}日", year, month_padded, day_padded));
            }
        }
    }

    // 住址提取: 住址(.+?)(:?公|民) 等
    let address_patterns = [
        r"住址(.+?)(?:公|民)",
        r"址(.+?)(?:公|民)",
        r"住(.+?)(?:公|民)",
    ];
    for pattern in &address_patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(&all_text) {
                if let Some(m) = caps.get(1) {
                    let address = m.as_str().trim();
                    if !address.is_empty() && address.len() >= 4 {
                        fields.address = Some(address.to_string());
                        break;
                    }
                }
            }
        }
    }

    // 身份证号提取
    if let Ok(re) = regex::Regex::new(r"\d{17}[\dXx]") {
        if let Some(caps) = re.captures(&all_text) {
            if let Some(m) = caps.get(0) {
                fields.id_number = Some(m.as_str().to_uppercase());
            }
        }
    }

    fields
}

/// 解析出生日期字符串
fn parse_birth_date(s: &str) -> Option<String> {
    // 尝试多种日期格式
    // 格式1: YYYY年MM月DD日
    if let Ok(re) = regex::Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})日") {
        if let Some(caps) = re.captures(s) {
            let year = caps.get(1)?.as_str();
            let month = caps.get(2)?.as_str();
            let day = caps.get(3)?.as_str();
            let month_padded = if month.len() == 1 { format!("0{}", month) } else { month.to_string() };
            let day_padded = if day.len() == 1 { format!("0{}", day) } else { day.to_string() };
            return Some(format!("{}年{}月{}日", year, month_padded, day_padded));
        }
    }

    // 格式2: YYYY-MM-DD 或 YYYY.MM.DD
    if let Ok(re) = regex::Regex::new(r"(\d{4})[\-\.](\d{1,2})[\-\.](\d{1,2})") {
        if let Some(caps) = re.captures(s) {
            let year = caps.get(1)?.as_str();
            let month = caps.get(2)?.as_str();
            let day = caps.get(3)?.as_str();
            let month_padded = if month.len() == 1 { format!("0{}", month) } else { month.to_string() };
            let day_padded = if day.len() == 1 { format!("0{}", day) } else { day.to_string() };
            return Some(format!("{}年{}月{}日", year, month_padded, day_padded));
        }
    }

    // 格式3: YYYYMMDD (纯数字)
    if let Ok(re) = regex::Regex::new(r"(\d{4})(\d{2})(\d{2})") {
        if let Some(caps) = re.captures(s) {
            let year = caps.get(1)?.as_str();
            let month = caps.get(2)?.as_str();
            let day = caps.get(3)?.as_str();
            return Some(format!("{}年{}月{}日", year, month, day));
        }
    }

    None
}

/// 身份证复印件生成 - 将正反面图片合成为PDF
#[tauri::command]
fn generate_idcard_copy(
    state: State<AppState>,
    front_image_base64: String,
    back_image_base64: String,
) -> Result<PdfGenerateResult, String> {
    // 第一步：矫正图片
    let (corrected_front, corrected_back) = {
        let corrector = state.corrector.lock().map_err(|e| e.to_string())?;
        
        if let Some(ref c) = *corrector {
            let front_result = c.process_image(&front_image_base64)
                .unwrap_or_else(|_| UnifiedCorrectionResult {
                    polygons: vec![],
                    bbox: vec![],
                    scores: vec![],
                    output_images: vec![front_image_base64.clone()],
                    labels: vec![],
                    layout: vec![],
                    center: vec![],
                });
            
            let back_result = c.process_image(&back_image_base64)
                .unwrap_or_else(|_| UnifiedCorrectionResult {
                    polygons: vec![],
                    bbox: vec![],
                    scores: vec![],
                    output_images: vec![back_image_base64.clone()],
                    labels: vec![],
                    layout: vec![],
                    center: vec![],
                });
            
            (
                front_result.output_images.first().cloned().unwrap_or(front_image_base64),
                back_result.output_images.first().cloned().unwrap_or(back_image_base64)
            )
        } else {
            (front_image_base64, back_image_base64)
        }
    };

    // 第二步：生成PDF
    match create_idcard_copy_pdf(&corrected_front, &corrected_back) {
        Ok(pdf_bytes) => {
            let pdf_base64 = general_purpose::STANDARD.encode(&pdf_bytes);
            Ok(PdfGenerateResult {
                success: true,
                message: "PDF生成成功".to_string(),
                pdf_base64: Some(pdf_base64),
            })
        }
        Err(e) => {
            Ok(PdfGenerateResult {
                success: false,
                message: format!("PDF生成失败: {}", e),
                pdf_base64: None,
            })
        }
    }
}

/// 创建身份证复印件PDF
fn create_idcard_copy_pdf(front_base64: &str, back_base64: &str) -> Result<Vec<u8>, String> {
    // 解码base64图片
    let front_bytes = general_purpose::STANDARD
        .decode(front_base64)
        .map_err(|e| format!("正面图片解码失败: {}", e))?;
    
    let back_bytes = general_purpose::STANDARD
        .decode(back_base64)
        .map_err(|e| format!("反面图片解码失败: {}", e))?;
    
    // 加载图片
    let front_img = image::load_from_memory(&front_bytes)
        .map_err(|e| format!("正面图片加载失败: {}", e))?;
    
    let back_img = image::load_from_memory(&back_bytes)
        .map_err(|e| format!("反面图片加载失败: {}", e))?;
    
    // 转换为RGB格式
    let front_rgb = front_img.to_rgb8();
    let back_rgb = back_img.to_rgb8();
    
    // 身份证标准尺寸: 85.6mm x 54mm
    // A4尺寸: 210mm x 297mm
    // 为了让身份证在A4纸上显示清晰，我们按实际尺寸放置
    
    let card_width_mm = 85.6;
    let card_height_mm = 54.0;
    let page_width = Mm(210.0);
    let page_height = Mm(297.0); // A4高度
    
    // 计算居中位置
    let margin_left = Mm((210.0 - card_width_mm) / 2.0);
    
    // 布局：正面在上方，反面在下方
    // PDF坐标系：原点在左下角，y向上
    // 上下边距相等：69.5mm
    // 正面（上方）：y = 297 - 69.5 - 54 = 173.5mm
    // 反面（下方）：y = 69.5mm
    let front_y = Mm(173.5);
    let back_y = Mm(69.5);
    
    // 创建PDF文档
    let (doc, page1, layer1) = PdfDocument::new(
        "身份证复印件",
        page_width,
        page_height,
        "Layer 1",
    );
    
    let current_layer = doc.get_page(page1).get_layer(layer1);
    
    // 添加正面图片
    add_image_to_layer(
        &current_layer,
        &front_rgb,
        front_rgb.width(),
        front_rgb.height(),
        margin_left,
        front_y,
        Mm(card_width_mm),
        Mm(card_height_mm),
    );
    
    // 添加反面图片
    add_image_to_layer(
        &current_layer,
        &back_rgb,
        back_rgb.width(),
        back_rgb.height(),
        margin_left,
        back_y,
        Mm(card_width_mm),
        Mm(card_height_mm),
    );
    
    // 保存PDF到内存
    let mut pdf_bytes = Vec::new();
    {
        let mut writer = BufWriter::new(&mut pdf_bytes);
        doc.save(&mut writer).map_err(|e| e.to_string())?;
    }
    
    Ok(pdf_bytes)
}

/// 统计已识别的字段数量
fn count_recognized_fields(fields: &IdCardFields) -> usize {
    let mut count = 0;
    if fields.name.is_some() { count += 1; }
    if fields.gender.is_some() { count += 1; }
    if fields.ethnicity.is_some() { count += 1; }
    if fields.birth_date.is_some() { count += 1; }
    if fields.address.is_some() { count += 1; }
    if fields.id_number.is_some() { count += 1; }
    count
}

/// 导出身份证识别结果到 Excel
/// 六个字段至少识别出三个才写入表格
#[tauri::command]
fn export_idcard_to_excel(
    results: Vec<RecognitionResult>,
) -> Result<ExcelExportResult, String> {
    if results.is_empty() {
        return Ok(ExcelExportResult {
            success: false,
            message: "没有可导出的数据".to_string(),
            excel_base64: None,
            total: 0,
            written: 0,
            skipped: 0,
        });
    }

    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();

    // 设置列宽
    worksheet.set_column_width(0, 30.0).map_err(|e| e.to_string())?; // 文件名
    worksheet.set_column_width(1, 25.0).map_err(|e| e.to_string())?; // 裁切后图片
    worksheet.set_column_width(2, 15.0).map_err(|e| e.to_string())?; // 姓名
    worksheet.set_column_width(3, 10.0).map_err(|e| e.to_string())?; // 性别
    worksheet.set_column_width(4, 12.0).map_err(|e| e.to_string())?; // 民族
    worksheet.set_column_width(5, 18.0).map_err(|e| e.to_string())?; // 出生日期
    worksheet.set_column_width(6, 40.0).map_err(|e| e.to_string())?; // 住址
    worksheet.set_column_width(7, 22.0).map_err(|e| e.to_string())?; // 身份证号

    // 设置行高（让图片能显示更大）
    worksheet.set_row_height(0, 25.0).map_err(|e| e.to_string())?;

    // 创建标题格式
    let header_format = Format::new()
        .set_bold()
        .set_align(FormatAlign::Center)
        .set_align(FormatAlign::VerticalCenter)
        .set_background_color(0x4472C4)
        .set_font_color(0xFFFFFF);

    // 创建居中对齐格式
    let center_format = Format::new()
        .set_align(FormatAlign::Center)
        .set_align(FormatAlign::VerticalCenter);

    // 创建左对齐格式
    let left_format = Format::new()
        .set_align(FormatAlign::Left)
        .set_align(FormatAlign::VerticalCenter);

    // 写入表头
    let headers = vec!["源文件名", "裁切后图片", "姓名", "性别", "民族", "出生日期", "住址", "身份证号"];
    for (col, header) in headers.iter().enumerate() {
        worksheet.write_string_with_format(0, col as u16, *header, &header_format).map_err(|e| e.to_string())?;
    }

    // 统计信息
    let mut row: u32 = 1;
    let mut written = 0;
    let mut skipped = 0;

    for result in &results {
        let recognized_count = count_recognized_fields(&result.fields);

        // 六个字段至少识别出三个才写入表格
        if recognized_count < 3 {
            skipped += 1;
            continue;
        }

        // 设置行高以便显示图片（100像素高度）
        worksheet.set_row_height(row, 100.0).map_err(|e| e.to_string())?;

        // 写入文件名
        worksheet.write_string_with_format(row, 0, &result.filename, &left_format).map_err(|e| e.to_string())?;

        // 插入裁切后的图片
        if !result.corrected_image.is_empty() {
            match insert_image_to_worksheet(worksheet, row, 1, &result.corrected_image) {
                Ok(_) => {},
                Err(e) => eprintln!("插入图片失败: {}", e),
            }
        }

        // 写入身份证字段
        worksheet.write_string_with_format(row, 2, result.fields.name.as_deref().unwrap_or(""), &center_format).map_err(|e| e.to_string())?;
        worksheet.write_string_with_format(row, 3, result.fields.gender.as_deref().unwrap_or(""), &center_format).map_err(|e| e.to_string())?;
        worksheet.write_string_with_format(row, 4, result.fields.ethnicity.as_deref().unwrap_or(""), &center_format).map_err(|e| e.to_string())?;
        worksheet.write_string_with_format(row, 5, result.fields.birth_date.as_deref().unwrap_or(""), &center_format).map_err(|e| e.to_string())?;
        worksheet.write_string_with_format(row, 6, result.fields.address.as_deref().unwrap_or(""), &left_format).map_err(|e| e.to_string())?;
        worksheet.write_string_with_format(row, 7, result.fields.id_number.as_deref().unwrap_or(""), &center_format).map_err(|e| e.to_string())?;

        row += 1;
        written += 1;
    }

    // 添加统计信息行
    let total_row = row + 1;
    let summary_format = Format::new()
        .set_bold()
        .set_align(FormatAlign::Left);

    worksheet.write_string_with_format(total_row, 0, &format!("总计: {} 条", results.len()), &summary_format).map_err(|e| e.to_string())?;
    worksheet.write_string_with_format(total_row + 1, 0, &format!("已写入: {} 条", written), &summary_format).map_err(|e| e.to_string())?;
    worksheet.write_string_with_format(total_row + 2, 0, &format!("已跳过(字段不足): {} 条", skipped), &summary_format).map_err(|e| e.to_string())?;

    // 保存到内存
    let mut buffer = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buffer);
        workbook.save_to_writer(&mut cursor)
            .map_err(|e| format!("保存Excel失败: {}", e))?;
    }

    let excel_base64 = general_purpose::STANDARD.encode(&buffer);

    Ok(ExcelExportResult {
        success: true,
        message: format!("导出成功! 总计: {} 条, 已写入: {} 条, 已跳过: {} 条", results.len(), written, skipped),
        excel_base64: Some(excel_base64),
        total: results.len(),
        written,
        skipped,
    })
}

/// 下载图片为 ZIP 压缩包
#[tauri::command]
fn download_images_as_zip(
    images: Vec<ImageItem>,
    options: ZipDownloadOptions,
) -> Result<ZipDownloadResult, String> {
    if images.is_empty() {
        return Ok(ZipDownloadResult {
            success: false,
            message: "没有可下载的图片".to_string(),
            zip_base64: None,
            filename: String::new(),
        });
    }

    // 创建内存中的 ZIP 写入器
    let mut buffer = Vec::new();
    {
        let mut zip_writer = zip::ZipWriter::new(Cursor::new(&mut buffer));
        let zip_options = zip::write::FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // 用于跟踪已使用的文件名，处理重名
        let mut used_filenames: HashSet<String> = HashSet::new();

        for (index, image_item) in images.iter().enumerate() {
            // 生成文件名
            let filename = generate_filename(
                &image_item.filename,
                image_item.fields.as_ref(),
                &options,
                &used_filenames,
                index,
            );
            used_filenames.insert(filename.clone());

            // 解码 base64 图片
            let image_data = general_purpose::STANDARD
                .decode(&image_item.image_base64)
                .map_err(|e| format!("图片 {} 解码失败: {}", image_item.filename, e))?;

            // 写入 ZIP
            zip_writer.start_file(&filename, zip_options)
                .map_err(|e| format!("添加文件 {} 到 ZIP 失败: {}", filename, e))?;
            zip_writer.write_all(&image_data)
                .map_err(|e| format!("写入文件 {} 失败: {}", filename, e))?;
        }

        // 完成 ZIP 写入
        zip_writer.finish()
            .map_err(|e| format!("完成 ZIP 文件失败: {}", e))?;
    }

    // 编码为 base64
    let zip_base64 = general_purpose::STANDARD.encode(&buffer);

    // 生成 ZIP 文件名
    let zip_filename = format!("证件图片_{}.zip", chrono::Local::now().format("%Y%m%d_%H%M%S"));

    Ok(ZipDownloadResult {
        success: true,
        message: format!("成功打包 {} 张图片", images.len()),
        zip_base64: Some(zip_base64),
        filename: zip_filename,
    })
}

/// 生成文件名
fn generate_filename(
    original_filename: &str,
    fields: Option<&IdCardFields>,
    options: &ZipDownloadOptions,
    used_filenames: &HashSet<String>,
    index: usize,
) -> String {
    // 获取原始文件扩展名
    let ext = original_filename
        .rsplit('.')
        .next()
        .map(|e| e.to_lowercase())
        .filter(|e| matches!(e.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp"))
        .unwrap_or_else(|| "jpg".to_string());

    let base_name = if options.auto_rename && fields.is_some() {
        let f = fields.unwrap();
        let mut parts = Vec::new();

        if options.include_name {
            if let Some(name) = &f.name {
                if !name.is_empty() {
                    parts.push(name.clone());
                }
            }
        }

        if options.include_id_number {
            if let Some(id) = &f.id_number {
                if !id.is_empty() {
                    parts.push(id.clone());
                }
            }
        }

        if parts.is_empty() {
            // 如果没有识别到信息，使用原始文件名（不含扩展名）
            original_filename
                .rsplit_once('.')
                .map(|(name, _)| name.to_string())
                .unwrap_or_else(|| original_filename.to_string())
        } else {
            parts.join("_")
        }
    } else {
        // 不重命名，使用原始文件名（不含扩展名）
        original_filename
            .rsplit_once('.')
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(|| original_filename.to_string())
    };

    // 清理文件名中的非法字符
    let safe_name: String = base_name
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect();

    // 处理重名：如果文件名已存在，添加序号
    let final_name = if used_filenames.contains(&format!("{}.{}" , safe_name, ext)) {
        let mut counter = 1;
        loop {
            let new_name = format!("{}_{}.{}", safe_name, counter, ext);
            if !used_filenames.contains(&new_name) {
                break new_name;
            }
            counter += 1;
            if counter > 9999 {
                // 防止无限循环，使用索引
                break format!("{}_{}.{}", safe_name, index, ext);
            }
        }
    } else {
        format!("{}.{}" , safe_name, ext)
    };

    final_name
}

/// 将 base64 图片插入到工作表，缩放至单元格大小
fn insert_image_to_worksheet(
    worksheet: &mut rust_xlsxwriter::Worksheet,
    row: u32,
    col: u16,
    image_base64: &str,
) -> Result<(), String> {
    // 解码 base64
    let image_data = general_purpose::STANDARD
        .decode(image_base64)
        .map_err(|e| format!("Base64 解码失败: {}", e))?;

    // 获取图片原始尺寸
    let (orig_width, orig_height) = image::load_from_memory(&image_data)
        .map(|img| (img.width(), img.height()))
        .unwrap_or((856, 540)); // 默认身份证比例

    // 目标单元格尺寸（像素）
    // 列宽 25 约等于 190 像素，行高 100 像素
    let cell_width = 180;
    let cell_height = 90;

    // 计算缩放比例，保持宽高比，填满单元格
    let scale_x = cell_width as f64 / orig_width as f64;
    let scale_y = cell_height as f64 / orig_height as f64;
    let scale = scale_x.min(scale_y);

    // 计算目标尺寸
    let target_width = (orig_width as f64 * scale) as u32;
    let target_height = (orig_height as f64 * scale) as u32;

    // 创建图片对象并设置大小
    let image = rust_xlsxwriter::Image::new_from_buffer(&image_data)
        .map_err(|e| format!("创建图片失败: {}", e))?
        .set_width(target_width)
        .set_height(target_height);

    // 插入图片
    worksheet.insert_image(row, col, &image)
        .map_err(|e| format!("插入图片失败: {}", e))?;

    Ok(())
}

/// 将图片添加到PDF图层
fn add_image_to_layer(
    layer: &PdfLayerReference,
    img_data: &image::RgbImage,
    img_width: u32,
    img_height: u32,
    x: Mm,
    y: Mm,
    target_width: Mm,
    target_height: Mm,
) {
    // 创建 ImageXObject
    let image_xobject = ImageXObject {
        width: Px(img_width as usize),
        height: Px(img_height as usize),
        color_space: ColorSpace::Rgb,
        bits_per_component: ColorBits::Bit8,
        interpolate: true,
        image_data: img_data.as_raw().clone(),
        image_filter: None,
        clipping_bbox: None,
    };
    
    // 创建Image对象
    let image_obj = Image::from(image_xobject);
    
    // 计算需要的 DPI 来使图片显示为目标尺寸
    // DPI = 像素数 / 英寸数
    // 英寸数 = 毫米数 / 25.4
    let target_width_inch = target_width.0 / 25.4;
    let target_height_inch = target_height.0 / 25.4;
    let dpi_x = (img_width as f64) / target_width_inch;
    let dpi_y = (img_height as f64) / target_height_inch;
    
    // 添加图片到图层
    image_obj.add_to_layer(
        layer.clone(),
        ImageTransform {
            translate_x: Some(x),
            translate_y: Some(y),
            dpi: Some(dpi_x), // 使用计算的 DPI
            scale_x: Some(1.0), // scale_x/y 在设置了 dpi 时会被忽略，但保留以防万一
            scale_y: Some(1.0),
            ..Default::default()
        },
    );
}

// ========== Utility Functions ==========

fn get_model_path() -> String {
    let exe_path = std::env::current_exe().expect("Failed to get exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe directory");

    let exe_dir_model = exe_dir.join("models").join("card_correction.onnx");
    if exe_dir_model.exists() {
        return exe_dir_model.to_string_lossy().to_string();
    }

    let models_dir = exe_dir.join("..").join("models");
    if models_dir.exists() {
        let model_path = models_dir.join("card_correction.onnx");
        if model_path.exists() {
            return model_path.to_string_lossy().to_string();
        }
    }

    // 尝试默认路径
    let default_model = std::path::Path::new(r#"D:\Downloads\IDcardOCR\idcard-ocr\models\card_correction.onnx"#);
    if default_model.exists() {
        return default_model.to_string_lossy().to_string();
    }

    let alt_model = std::path::Path::new(r#"D:\Downloads\IDcardOCR\models\card_correction.onnx"#);
    if alt_model.exists() {
        return alt_model.to_string_lossy().to_string();
    }

    r#"D:\Downloads\IDcardOCR\idcard-ocr\models\card_correction.onnx"#.to_string()
}

fn get_dll_path() -> String {
    let exe_path = std::env::current_exe().expect("Failed to get exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe directory");

    let possible_dll_paths = [
        exe_dir.join("card_correction.dll"),
        exe_dir.join("..").join("card_correction.dll"),
        exe_dir.join("..").join("..").join("card_correction.dll"),
        std::path::Path::new(r#"D:\Downloads\IDcardOCR\idcard-ocr\src-tauri\card_correction.dll"#).to_path_buf(),
        std::path::Path::new(r#"D:\Downloads\IDcardOCR\models\card_correction.dll"#).to_path_buf(),
    ];

    for path in &possible_dll_paths {
        if path.exists() {
            return path.to_string_lossy().to_string();
        }
    }

    String::new()
}

fn get_ocr_models_path() -> String {
    let exe_path = std::env::current_exe().expect("Failed to get exe path");
    let exe_dir = exe_path.parent().expect("Failed to get exe directory");

    // 检查 exe 同级 models 目录
    let exe_dir_models = exe_dir.join("models");
    if exe_dir_models.join("ch_PP-OCRv5_mobile_det.onnx").exists() {
        return exe_dir_models.to_string_lossy().to_string();
    }

    // 检查上级 models 目录
    let parent_models = exe_dir.join("..").join("models");
    if parent_models.join("ch_PP-OCRv5_mobile_det.onnx").exists() {
        return parent_models.to_string_lossy().to_string();
    }

    // 检查 src-tauri/models 目录
    let src_tauri_models = exe_dir.join("..").join("src-tauri").join("models");
    if src_tauri_models.join("ch_PP-OCRv5_mobile_det.onnx").exists() {
        return src_tauri_models.to_string_lossy().to_string();
    }

    // 默认路径
    let default_path = std::path::Path::new(r"D:\Downloads\IDcardOCR\idcard-ocr\src-tauri\models");
    if default_path.join("ch_PP-OCRv5_mobile_det.onnx").exists() {
        return default_path.to_string_lossy().to_string();
    }

    // 返回默认路径
    r"D:\Downloads\IDcardOCR\idcard-ocr\src-tauri\models".to_string()
}

fn create_corrector() -> Result<DllCorrector, String> {
    let dll_path = get_dll_path();
    if dll_path.is_empty() {
        return Err("card_correction.dll not found!".to_string());
    }

    let model_path = get_model_path();
    DllCorrector::new(&dll_path, &model_path)
}

fn main() {
    println!("========================================");
    println!("        证件工具箱                      ");
    println!("========================================");

    // 初始化矫正器
    let corrector = match create_corrector() {
        Ok(_) => {
            println!("✅ 矫正器加载成功");
            Some(DllCorrector::new(&get_dll_path(), &get_model_path()).unwrap())
        }
        Err(e) => {
            eprintln!("⚠️ 矫正器加载失败: {}", e);
            None
        }
    };

    // 初始化 OCR 引擎
    let models_dir = get_ocr_models_path();
    let ocr_engine = match OcrEngine::new(&models_dir) {
        Ok(engine) => {
            println!("✅ OCR引擎加载成功");
            Some(engine)
        }
        Err(e) => {
            eprintln!("⚠️ OCR引擎加载失败: {}", e);
            None
        }
    };

    println!("========================================");

    tauri::Builder::default()
        .manage(AppState {
            corrector: Mutex::new(corrector),
            ocr_engine: Mutex::new(ocr_engine),
        })
        .invoke_handler(tauri::generate_handler![
            check_ocr_status,
            correct_image,
            recognize_idcard,
            batch_recognize_idcard,
            ocr_image,
            generate_idcard_copy,
            export_idcard_to_excel,
            download_images_as_zip,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
