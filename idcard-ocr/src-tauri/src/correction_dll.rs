use std::ffi::{c_char, c_int, c_uchar, c_void, CStr, CString};
use std::ptr::null_mut;
use std::sync::Arc;
use libloading::{Library, Symbol};

// 错误码定义
pub const CARD_SUCCESS: c_int = 0;

/// 检测结果结构体（必须与C++完全一致）
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CardDetection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub x3: f32,
    pub y3: f32,
    pub score: f32,
    pub angle: c_int,
    pub cx: f32,
    pub cy: f32,
    pub ftype: c_int,
}

/// 矫正结果结构体（必须与C++完全一致）
#[repr(C)]
pub struct CardCorrectionResult {
    pub detections: *mut CardDetection,
    pub num_detections: c_int,
    pub output_images: *mut *mut c_uchar,
    pub output_image_sizes: *mut c_int,
    pub output_image_angles: *mut c_int,
    pub num_output_images: c_int,
}

/// 模型句柄类型
pub type CardCorrectionHandle = *mut c_void;

// DLL函数类型定义
type CardCorrectionCreateFn = unsafe extern "C" fn(*const c_char) -> CardCorrectionHandle;
type CardCorrectionDestroyFn = unsafe extern "C" fn(CardCorrectionHandle);
type CardCorrectionInferFn = unsafe extern "C" fn(
    CardCorrectionHandle,
    *const c_uchar,
    c_int,
    c_int,
    c_int,
    *mut *mut CardCorrectionResult,
) -> c_int;
type CardCorrectionFreeResultFn = unsafe extern "C" fn(*mut CardCorrectionResult);
type CardCorrectionGetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;

/// 检测结果（Rust安全版本）
#[derive(Debug, Clone)]
pub struct Detection {
    pub polygon: Vec<[f32; 2]>,
    pub score: f32,
    pub angle: i32,
    pub center: [f32; 2],
    pub ftype: i32,
}

/// 输出图像
#[derive(Debug, Clone)]
pub struct OutputImage {
    pub data: Vec<u8>,
    pub angle: i32,
}

/// 完整的矫正结果
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    pub detections: Vec<Detection>,
    pub output_images: Vec<OutputImage>,
}

/// DLL加载器
pub struct CardCorrectionDll {
    lib: Arc<Library>,
}

impl CardCorrectionDll {
    /// 加载DLL
    pub fn new(dll_path: &str) -> Result<Self, String> {
        unsafe {
            let lib = Library::new(dll_path)
                .map_err(|e| format!("Failed to load DLL: {}", e))?;
            Ok(CardCorrectionDll { lib: Arc::new(lib) })
        }
    }
}

/// 模型实例
pub struct CardCorrectionModel {
    handle: CardCorrectionHandle,
    lib: Arc<Library>,
}

unsafe impl Send for CardCorrectionModel {}
unsafe impl Sync for CardCorrectionModel {}

impl Drop for CardCorrectionModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let destroy_fn: Symbol<CardCorrectionDestroyFn> = 
                    self.lib.get(b"card_correction_destroy").unwrap();
                destroy_fn(self.handle);
            }
        }
    }
}

impl CardCorrectionModel {
    /// 创建模型实例
    pub fn new(dll: &CardCorrectionDll, model_path: &str) -> Result<Self, String> {
        let c_path = CString::new(model_path)
            .map_err(|e| format!("Invalid model path: {}", e))?;

        unsafe {
            let create_fn: Symbol<CardCorrectionCreateFn> = 
                dll.lib.get(b"card_correction_create")
                    .map_err(|e| format!("Failed to load function: {}", e))?;

            let handle = create_fn(c_path.as_ptr());
            if handle.is_null() {
                return Err("Failed to create model".to_string());
            }

            Ok(CardCorrectionModel {
                handle,
                lib: dll.lib.clone(),
            })
        }
    }

    /// 执行推理
    pub fn infer(&self, image_data: &[u8], width: i32, height: i32, channels: i32) -> Result<CorrectionResult, String> {
        if image_data.len() < (width * height * channels) as usize {
            return Err("Image data size mismatch".to_string());
        }

        let mut result_ptr: *mut CardCorrectionResult = null_mut();

        unsafe {
            let infer_fn: Symbol<CardCorrectionInferFn> = 
                self.lib.get(b"card_correction_infer")
                    .map_err(|e| format!("Failed to load function: {}", e))?;

            let ret = infer_fn(
                self.handle,
                image_data.as_ptr(),
                width,
                height,
                channels,
                &mut result_ptr,
            );

            if ret != CARD_SUCCESS {
                let get_error_fn: Symbol<CardCorrectionGetErrorStringFn> =
                    self.lib.get(b"card_correction_get_error_string").unwrap();
                let error_msg = get_error_fn(ret);
                let error_str = if error_msg.is_null() {
                    format!("Error code: {}", ret)
                } else {
                    CStr::from_ptr(error_msg).to_string_lossy().into_owned()
                };
                return Err(format!("Inference failed: {}", error_str));
            }

            if result_ptr.is_null() {
                return Err("Result pointer is null".to_string());
            }

            // 提取结果
            let result = self.extract_result(result_ptr);

            // 释放内存
            let free_fn: Symbol<CardCorrectionFreeResultFn> =
                self.lib.get(b"card_correction_free_result").unwrap();
            free_fn(result_ptr);

            Ok(result)
        }
    }

    /// 安全提取结果
    unsafe fn extract_result(&self, result_ptr: *mut CardCorrectionResult) -> CorrectionResult {
        let raw_result = &*result_ptr;

        // 提取检测结果
        let mut detections = Vec::new();
        if !raw_result.detections.is_null() && raw_result.num_detections > 0 {
            let det_slice = std::slice::from_raw_parts(
                raw_result.detections,
                raw_result.num_detections as usize
            );

            for det in det_slice {
                detections.push(Detection {
                    polygon: vec![
                        [det.x0, det.y0],
                        [det.x1, det.y1],
                        [det.x2, det.y2],
                        [det.x3, det.y3],
                    ],
                    score: det.score,
                    angle: det.angle,
                    center: [det.cx, det.cy],
                    ftype: det.ftype,
                });
            }
        }

        // 提取输出图像
        let mut output_images = Vec::new();
        if !raw_result.output_images.is_null() && raw_result.num_output_images > 0 {
            let num_images = raw_result.num_output_images as usize;
            let img_ptrs = std::slice::from_raw_parts(
                raw_result.output_images,
                num_images
            );
            let img_sizes = std::slice::from_raw_parts(
                raw_result.output_image_sizes,
                num_images
            );
            let img_angles = std::slice::from_raw_parts(
                raw_result.output_image_angles,
                num_images
            );

            for i in 0..num_images {
                if !img_ptrs[i].is_null() && img_sizes[i] > 0 {
                    let img_data = std::slice::from_raw_parts(img_ptrs[i], img_sizes[i] as usize);
                    output_images.push(OutputImage {
                        data: img_data.to_vec(),
                        angle: img_angles[i],
                    });
                }
            }
        }

        CorrectionResult {
            detections,
            output_images,
        }
    }

    /// 处理Base64编码的图像
    pub fn process_base64(&self, image_base64: &str) -> Result<CorrectionResult, String> {
        use base64::{engine::general_purpose, Engine};

        // 解码base64
        let image_data = general_purpose::STANDARD
            .decode(image_base64)
            .map_err(|e| format!("Failed to decode base64: {}", e))?;

        // 加载图像
        let img = image::load_from_memory(&image_data)
            .map_err(|e| format!("Failed to load image: {}", e))?;

        let (width, height) = (img.width() as i32, img.height() as i32);

        // 转换为BGR格式
        let rgb_img = img.to_rgb8();
        let mut bgr_data = Vec::with_capacity((width * height * 3) as usize);

        for pixel in rgb_img.pixels() {
            bgr_data.push(pixel[2]); // B
            bgr_data.push(pixel[1]); // G
            bgr_data.push(pixel[0]); // R
        }

        self.infer(&bgr_data, width, height, 3)
    }
}
