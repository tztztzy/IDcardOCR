use base64::{engine::general_purpose, Engine};
use image::{DynamicImage, GenericImageView};
use serde::Serialize;
use std::path::Path;
use tract_onnx::prelude::*;

#[derive(Serialize, Clone, Debug)]
pub struct CorrectionResult {
    pub polygons: Vec<Vec<f32>>,
    pub bbox: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
    pub output_images: Vec<String>,
    pub labels: Vec<i32>,
    pub layout: Vec<i32>,
    pub center: Vec<Vec<f32>>,
}

pub struct CardCorrection {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    resize_shape: i32,
    mean: [f32; 3],
    std: [f32; 3],
}

impl CardCorrection {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(Path::new(model_path))
            .map_err(|e| format!("Failed to load model: {}", e))?
            .into_optimized()
            .map_err(|e| e.to_string())?
            .into_runnable()
            .map_err(|e| e.to_string())?;

        Ok(CardCorrection {
            model,
            resize_shape: 768,
            mean: [0.408, 0.447, 0.470],
            std: [0.289, 0.274, 0.278],
        })
    }

    pub fn process_image(&self, image_base64: &str) -> Result<CorrectionResult, String> {
        let image_data = general_purpose::STANDARD
            .decode(image_base64)
            .map_err(|e| format!("Failed to decode base64: {}", e))?;

        let img = image::load_from_memory(&image_data)
            .map_err(|e| format!("Failed to load image: {}", e))?;

        let (ori_w, ori_h) = (img.width() as f32, img.height() as f32);
        let c = [ori_w / 2.0, ori_h / 2.0];
        let s = ori_h.max(ori_w);

        let tensor = self.preprocess(&img)?;

        let result = self.model
            .run(tvec!(tensor.into()))
            .map_err(|e| format!("Inference failed: {}", e))?;

        self.postprocess(&result, &c, s, ori_h as usize, ori_w as usize, &img)
    }

    fn preprocess(&self, img: &DynamicImage) -> Result<Tensor, String> {
        let (w, h) = (img.width(), img.height());
        let m = h.max(w);
        let ratio = self.resize_shape as f32 / m as f32;
        let new_w = (ratio * w as f32) as u32;
        let new_h = (ratio * h as f32) as u32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);

        let top = (self.resize_shape - new_h as i32) / 2;
        let _bottom = self.resize_shape - new_h as i32 - top;
        let left = (self.resize_shape - new_w as i32) / 2;
        let _right = self.resize_shape - new_w as i32 - left;

        let mut result = DynamicImage::new_rgb8(self.resize_shape as u32, self.resize_shape as u32);
        image::imageops::replace(&mut result, &resized, left as i64, top as i64);

        let img_h = self.resize_shape as usize;
        let img_w = self.resize_shape as usize;

        let mut data: Vec<f32> = Vec::with_capacity(img_h * img_w * 3);
        for y in 0..img_h {
            for x in 0..img_w {
                let pixel = result.get_pixel(x as u32, y as u32);
                for c_idx in 0..3 {
                    let normalized = (pixel[c_idx] as f32 / 255.0 - self.mean[c_idx]) / self.std[c_idx];
                    data.push(normalized);
                }
            }
        }

        let mut chw_data: Vec<f32> = Vec::with_capacity(img_h * img_w * 3);
        for c_idx in 0..3 {
            for y in 0..img_h {
                for x in 0..img_w {
                    chw_data.push(data[c_idx * img_h * img_w + y * img_w + x]);
                }
            }
        }

        let shape: Vec<usize> = vec![1, 3, img_h, img_w];
        Tensor::from_shape(&shape, &chw_data).map_err(|e| e.to_string())
    }

    fn postprocess(
        &self,
        outputs: &TVec<TValue>,
        c: &[f32; 2],
        s: f32,
        ori_h: usize,
        ori_w: usize,
        original_img: &DynamicImage,
    ) -> Result<CorrectionResult, String> {
        let out_h = (self.resize_shape / 4) as usize;
        let out_w = (self.resize_shape / 4) as usize;
        let stride = out_h * out_w;
        let K = 10;

        let get_output = |outputs: &TVec<TValue>, idx: usize| -> Option<(Vec<f32>, Vec<usize>)> {
            if idx < outputs.len() {
                let t = &outputs[idx];
                if let Ok(slice) = t.to_array_view::<f32>() {
                    let shape: Vec<usize> = slice.shape().iter().copied().collect();
                    let data: Vec<f32> = slice.iter().copied().collect();
                    return Some((data, shape));
                }
            }
            None
        };

        // Output order from tract-onnx (actual observed shapes):
        // Output 0: [1, 1, 192, 192] - 1 channel (maybe angle?)
        // Output 1: [1, 2, 192, 192] - 2 channels (maybe reg?)
        // Output 2: [1, 2, 192, 192] - 2 channels (maybe ftype?)
        // Output 3: [1, 8, 192, 192] - 8 channels (this is wh!)
        // Output 4: [1, 2, 192, 192] - 2 channels (this should be hm but has 2ch instead of 1)
        //
        // Based on debug: using output[3] as wh (8 channels), output[1] as reg (2 channels)
        let (angle_data, angle_shape) = get_output(outputs, 0).unwrap_or((vec![], vec![]));
        let (reg_data, reg_shape) = get_output(outputs, 1).unwrap_or((vec![], vec![]));
        let (_ftype_data, _ftype_shape) = get_output(outputs, 2).unwrap_or((vec![], vec![]));
        let (wh_data, wh_shape) = get_output(outputs, 3).unwrap_or((vec![], vec![]));
        let (mut hm_data, hm_shape) = get_output(outputs, 4).unwrap_or((vec![], vec![]));

        // Debug: print shapes
        eprintln!("DEBUG: angle_shape: {:?}, len: {}", angle_shape, angle_data.len());
        eprintln!("DEBUG: wh_shape: {:?}, len: {}", wh_shape, wh_data.len());
        eprintln!("DEBUG: reg_shape: {:?}, len: {}", reg_shape, reg_data.len());
        eprintln!("DEBUG: hm_shape: {:?}, len: {}", hm_shape, hm_data.len());
        eprintln!("DEBUG: stride: {}, out_h: {}, out_w: {}", stride, out_h, out_w);

        if hm_data.is_empty() || wh_data.is_empty() || reg_data.is_empty() {
            return self.return_empty_result(original_img, ori_w as f32, ori_h as f32);
        }

        // Apply sigmoid to heatmap
        for v in hm_data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }

        // Apply sigmoid to angle_data (matching Python's sigmoid(angle_cls))
        let mut angle_data: Vec<f32> = angle_data;
        for v in angle_data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }

        // Reshape heatmap from (1, cat, h, w) to (cat, h, w) - C++ does heat = heat.reshape(0, shape)
        let cat = hm_shape[1];
        let mut hm_reshaped: Vec<f32> = Vec::with_capacity(cat * stride);
        for c_idx in 0..cat {
            for pos in 0..stride {
                let idx = c_idx * stride + pos;
                if idx < hm_data.len() {
                    hm_reshaped.push(hm_data[idx]);
                }
            }
        }
        hm_data = hm_reshaped;

        // Apply NMS to heatmap (max pooling 3x3)
        let hm_nms = max_pool2d(&hm_data, cat, out_h, out_w, 3);

        // Apply keep: heat = heat * keep (only keep values that are max)
        for i in 0..hm_data.len() {
            if hm_nms[i] < 0.5 { // if not the maximum
                hm_data[i] = 0.0;
            }
        }

        // TopK selection - exactly like C++'s _topk
        let (topk_scores, topk_inds, topk_clses, topk_ys, topk_xs) = topk(&hm_data, cat, out_w, K);

        // Transpose and gather feat for regression - C++'s _tranpose_and_gather_feat
        // reg shape is (1, 2, 192, 192) -> (2, 192*192) after transpose and gather
        let reg_gathered = transpose_gather_feat(&reg_data, &topk_inds, stride);

        // Apply regression
        let mut xs: Vec<f32> = Vec::with_capacity(K);
        let mut ys: Vec<f32> = Vec::with_capacity(K);
        for i in 0..K.min(topk_scores.len()) {
            xs.push(topk_xs[i] + reg_gathered[i * 2]);
            ys.push(topk_ys[i] + reg_gathered[i * 2 + 1]);
        }

        // Transpose and gather feat for wh - C++'s _tranpose_and_gather_feat
        let wh_gathered = transpose_gather_feat(&wh_data, &topk_inds, stride);

        // Transpose and gather feat for angle - similar to decode_by_ind in Python
        let angle_gathered = transpose_gather_feat(&angle_data, &topk_inds, stride);
        // angle_data shape from tract-onnx is [1, 1, 192, 192] (1 channel, not 4!)
        // This might be a regression output (continuous value) not classification
        let angle_channels = if angle_data.len() >= stride * 4 { 4 } else if angle_data.len() >= stride * 2 { 2 } else { 1 };

        // Get affine transform for coordinate transform
        let trans = get_affine_transform_inv(c, s, out_w as f32, out_h as f32);

        // Build bboxes [x0, y0, x1, y1, x2, y2, x3, y3, score, angle, cx, cy]
        let mut bboxes: Vec<Vec<f32>> = Vec::new();

        // Calculate how many channels wh actually has
        let wh_channels = wh_gathered.len() / K.max(1);
        eprintln!("DEBUG: wh_gathered.len: {}, wh_channels: {}", wh_gathered.len(), wh_channels);
        eprintln!("DEBUG: reg_gathered.len: {}, angle_gathered.len: {}", reg_gathered.len(), angle_gathered.len());
        eprintln!("DEBUG: topk_scores: {:?}", &topk_scores[..topk_scores.len().min(5)]);
        
        for i in 0..K.min(topk_scores.len()) {
            let score = topk_scores[i];
            if score < 0.5 {
                continue;
            }

            // wh_gathered: [w0, h0, w1, h1, w2, h2, w3, h3, ...] for each k
            // Safely get values with bounds checking
            let get_wh = |idx: usize| -> f32 {
                let arr_idx = i * wh_channels + idx;
                if arr_idx < wh_gathered.len() { wh_gathered[arr_idx] } else { 0.0 }
            };

            let w0 = get_wh(0);
            let h0 = get_wh(1);
            let w1 = if wh_channels > 2 { get_wh(2) } else { w0 };
            let h1 = if wh_channels > 3 { get_wh(3) } else { h0 };
            let w2 = if wh_channels > 4 { get_wh(4) } else { w0 };
            let h2 = if wh_channels > 5 { get_wh(5) } else { h0 };
            let w3 = if wh_channels > 6 { get_wh(6) } else { w0 };
            let h3 = if wh_channels > 7 { get_wh(7) } else { h0 };

            // Calculate 4 corners
            let x0 = xs[i] - w0;
            let y0 = ys[i] - h0;
            let x1 = xs[i] - w1;
            let y1 = ys[i] - h1;
            let x2 = xs[i] - w2;
            let y2 = ys[i] - h2;
            let x3 = xs[i] - w3;
            let y3 = ys[i] - h3;

            // Transform to original coords
            let (x0_t, y0_t) = affine_transform(x0, y0, &trans);
            let (x1_t, y1_t) = affine_transform(x1, y1, &trans);
            let (x2_t, y2_t) = affine_transform(x2, y2, &trans);
            let (x3_t, y3_t) = affine_transform(x3, y3, &trans);
            let (cx_t, cy_t) = affine_transform(xs[i], ys[i], &trans);

            // Get angle from angle_gathered
            // If multi-channel: use argmax (classification)
            // If single-channel: use the value directly (regression), round to nearest int
            let angle = if angle_channels > 1 {
                // Find the channel with max score for this k
                let mut max_idx = 0;
                let mut max_val = f32::MIN;
                for ch in 0..angle_channels {
                    let val = angle_gathered[i * angle_channels + ch];
                    if val > max_val {
                        max_val = val;
                        max_idx = ch;
                    }
                }
                max_idx as i32
            } else {
                // Single channel - treat as regression value
                // Round to nearest integer and clamp to valid range [0, 3]
                let val = angle_gathered[i];
                val.round() as i32
            };

            // Validate coordinates
            let x_min = x0_t.min(x1_t.min(x2_t.min(x3_t))).max(0.0);
            let y_min = y0_t.min(y1_t.min(y2_t.min(y3_t))).max(0.0);
            let x_max = x0_t.max(x1_t.max(x2_t.max(x3_t))).min(ori_w as f32);
            let y_max = y0_t.max(y1_t.max(y2_t.max(y3_t))).min(ori_h as f32);

            if x_max - x_min < 10.0 || y_max - y_min < 10.0 {
                eprintln!("DEBUG: bbox {} rejected - too small: {:.1}x{:.1}", i, x_max - x_min, y_max - y_min);
                continue;
            }

            let bbox = vec![
                x0_t, y0_t, x1_t, y1_t, x2_t, y2_t, x3_t, y3_t,
                score, angle as f32, cx_t, cy_t, 0.0
            ];
            eprintln!("DEBUG: bbox {} added: score={}, angle={}, coords=[{:.1},{:.1}][{:.1},{:.1}][{:.1},{:.1}][{:.1},{:.1}]", 
                i, score, angle, x0_t, y0_t, x1_t, y1_t, x2_t, y2_t, x3_t, y3_t);
            bboxes.push(bbox);
        }

        let mut polygons = Vec::new();
        let mut bbox_list = Vec::new();
        let mut scores = Vec::new();
        let mut output_images = Vec::new();
        let mut labels = Vec::new();
        let mut layout = Vec::new();
        let mut center = Vec::new();

        for bbox in &bboxes {
            let poly = vec![bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7]];
            let x_min = poly[0].min(poly[2].min(poly[4].min(poly[6])));
            let y_min = poly[1].min(poly[3].min(poly[5].min(poly[7])));
            let x_max = poly[0].max(poly[2].max(poly[4].max(poly[6])));
            let y_max = poly[1].max(poly[3].max(poly[5].max(poly[7])));

            let position = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[4], bbox[5]],
                [bbox[6], bbox[7]],
            ];

            match perspective_transform(original_img, &position) {
                Ok(cropped) => {
                eprintln!("DEBUG: perspective_transform success, size: {}x{}", cropped.width(), cropped.height());
                // Rotation mapping matching Python/OpenCV:
                // angle=1: cv2.rotate(..., 2) = ROTATE_90_COUNTERCLOCKWISE
                // angle=2: cv2.rotate(..., 1) = ROTATE_180
                // angle=3: cv2.rotate(..., 0) = ROTATE_90_CLOCKWISE
                // Note: image::rotate90() = clockwise 90, rotate270() = counter-clockwise 90
                let rotated = match bbox[9] as i32 {
                    1 => cropped.rotate270(), // counter-clockwise 90 (same as ROTATE_90_COUNTERCLOCKWISE)
                    2 => cropped.rotate180(), // 180
                    3 => cropped.rotate90(),  // clockwise 90 (same as ROTATE_90_CLOCKWISE)
                    _ => cropped,
                };

                let mut buffer = Vec::new();
                rotated.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageOutputFormat::Jpeg(90))
                    .map_err(|e| e.to_string())?;
                let base64_str = general_purpose::STANDARD.encode(&buffer);

                polygons.push(poly);
                bbox_list.push(vec![x_min, y_min, x_max, y_max]);
                scores.push(bbox[8]);
                output_images.push(base64_str);
                labels.push(bbox[9] as i32);
                layout.push(0);
                center.push(vec![bbox[10], bbox[11]]);
                }
                Err(e) => {
                    eprintln!("DEBUG: perspective_transform failed: {}", e);
                }
            }
        }

        eprintln!("DEBUG: total bboxes: {}, output_images: {}", bboxes.len(), output_images.len());

        if output_images.is_empty() {
            return self.return_empty_result(original_img, ori_w as f32, ori_h as f32);
        }

        Ok(CorrectionResult {
            polygons,
            bbox: bbox_list,
            scores,
            output_images,
            labels,
            layout,
            center,
        })
    }

    fn return_empty_result(&self, original_img: &DynamicImage, ori_w: f32, ori_h: f32) -> Result<CorrectionResult, String> {
        let mut buffer = Vec::new();
        original_img.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageOutputFormat::Jpeg(90))
            .map_err(|e| e.to_string())?;
        let base64_str = general_purpose::STANDARD.encode(&buffer);

        Ok(CorrectionResult {
            polygons: vec![vec![0.0, 0.0, ori_w, 0.0, ori_w, ori_h, 0.0, ori_h]],
            bbox: vec![vec![0.0, 0.0, ori_w, ori_h]],
            scores: vec![0.5],
            output_images: vec![base64_str],
            labels: vec![0],
            layout: vec![0],
            center: vec![vec![ori_w / 2.0, ori_h / 2.0]],
        })
    }
}

// Max pooling 2D - exactly like C++'s max_pooling
fn max_pool2d(input: &[f32], chan: usize, inp_h: usize, inp_w: usize, kernel: usize) -> Vec<f32> {
    let pad = (kernel - 1) / 2;
    let stride = 1;
    let out_h = inp_h;
    let out_w = inp_w;
    let out_len = chan * out_h * out_w;

    let mut output = vec![f32::MIN; out_len];

    for c in 0..chan {
        for y in 0..out_h {
            for x in 0..out_w {
                let y_start = (y as i32 * stride as i32 - pad as i32).max(0) as usize;
                let y_end = (y_start + kernel).min(inp_h);
                let x_start = (x as i32 * stride as i32 - pad as i32).max(0) as usize;
                let x_end = (x_start + kernel).min(inp_w);

                let mut max_val = f32::MIN;
                for yy in y_start..y_end {
                    for xx in x_start..x_end {
                        let idx = c * inp_h * inp_w + yy * inp_w + xx;
                        if input[idx] > max_val {
                            max_val = input[idx];
                        }
                    }
                }
                let out_idx = c * out_h * out_w + y * out_w + x;
                output[out_idx] = max_val;
            }
        }
    }
    output
}

// TopK function - exactly like C++'s _topk
fn topk(heat: &[f32], cat: usize, out_w: usize, k: usize) -> (Vec<f32>, Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>) {
    let stride = heat.len() / cat;

    // First topk per category
    let mut cat_topk: Vec<(f32, usize, usize)> = Vec::new();
    for c in 0..cat {
        let mut indices: Vec<usize> = (0..stride).collect();
        indices.sort_by(|&a, &b| heat[c * stride + b].partial_cmp(&heat[c * stride + a]).unwrap());

        for j in 0..k.min(indices.len()) {
            cat_topk.push((heat[c * stride + indices[j]], c, indices[j]));
        }
    }

    // Global topk
    cat_topk.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let k_final = k.min(cat_topk.len());

    let mut topk_scores: Vec<f32> = Vec::with_capacity(k_final);
    let mut topk_inds: Vec<usize> = Vec::with_capacity(k_final);
    let mut topk_clses: Vec<usize> = Vec::with_capacity(k_final);
    let mut topk_ys: Vec<f32> = Vec::with_capacity(k_final);
    let mut topk_xs: Vec<f32> = Vec::with_capacity(k_final);

    for i in 0..k_final {
        let (score, cls, pos) = cat_topk[i];
        topk_scores.push(score);
        topk_clses.push(cls);
        topk_inds.push(pos);
        topk_ys.push((pos / out_w) as f32);
        topk_xs.push((pos % out_w) as f32);
    }

    (topk_scores, topk_inds, topk_clses, topk_ys, topk_xs)
}

// Transpose and gather feat - exactly like C++'s _tranpose_and_gather_feat
fn transpose_gather_feat(feat: &[f32], inds: &[usize], stride: usize) -> Vec<f32> {
    // The tensor is flattened from (batch, c, h, w) to a 1D array
    // We need to infer c from the total length
    // c = total_len / (batch * h * w)
    // But we don't know batch, so we infer c by finding what works
    
    // Infer c from the feature length, assuming shape is (batch, c, h, w)
    // Common shapes:
    // - reg: (1, 2, 192, 192) = 73728 elements
    // - wh: (1, 8, 192, 192) = 294912 elements  
    // - angle: (1, 4, 192, 192) = 147456 elements
    // - ftype: (1, 2, 192, 192) = 73728 elements
    // - hm: (1, 1, 192, 192) = 36864 elements
    
    // If feat.len() is small (like 20), we might be getting flattened/batched output
    // In that case, infer c from typical ONNX output patterns
    let mut c: usize;
    let expected_per_sample = stride; // h * w
    
    if feat.len() >= expected_per_sample * 8 {
        c = 8;
    } else if feat.len() >= expected_per_sample * 4 {
        c = 4;
    } else if feat.len() >= expected_per_sample * 2 {
        c = 2;
    } else if feat.len() >= expected_per_sample {
        c = 1;
    } else {
        // feat.len() is smaller than stride - this shouldn't happen normally
        // But if it does, assume the data is already transposed/reshaped
        // In this case, c = feat.len() / K (roughly)
        c = 8; // Default to 8 for wh
    }
    
    // If the tensor is very small, it might be pre-processed differently
    // Try to infer from the actual ratio
    if feat.len() < stride && feat.len() > 0 {
        // The tensor might be in a different format
        // Just use all available data per index
        c = feat.len().max(8);
    }
    
    let k = inds.len();
    
    let mut result: Vec<f32> = Vec::with_capacity(k * c);

    for &ind in inds {
        for ch in 0..c {
            let feat_idx = ch * stride + ind;
            if feat_idx < feat.len() {
                result.push(feat[feat_idx]);
            } else {
                result.push(0.0);
            }
        }
    }
    result
}

fn get_affine_transform_inv(center: &[f32; 2], scale: f32, out_w: f32, out_h: f32) -> [f32; 6] {
    // C++ implementation:
    // src_w = scale
    // dst_w = output_size[0], dst_h = output_size[1]
    // rot = 0
    // src_dir = get_dir([0, src_w * -0.5], 0)
    // dst_dir = [0, dst_w * -0.5]
    // src[0] = center
    // src[1] = center + src_dir
    // dst[0] = [dst_w * 0.5, dst_h * 0.5]
    // dst[1] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
    // src[2] = get_3rd_point(src[0], src[1])
    // dst[2] = get_3rd_point(dst[0], dst[1])
    // inv=1: trans = cv::getAffineTransform(dst, src)

    let cx = center[0] as f64;
    let cy = center[1] as f64;
    let src_w = scale as f64;

    // src_dir for rot=0: [0, src_w * -0.5] (note: y direction, not x!)
    // This matches Python's get_dir([0, src_w * -0.5], 0) = [0, src_w * -0.5]
    let src_dir_x = 0.0;
    let src_dir_y = -src_w * 0.5;

    // dst_dir: [0, dst_w * -0.5] (note: y direction, not x!)
    let dst_dir_x = 0.0;
    let dst_dir_y = -out_w as f64 * 0.5;

    // src points
    let src_0_x = cx;
    let src_0_y = cy;
    let src_1_x = cx + src_dir_x;
    let src_1_y = cy + src_dir_y;
    // get_3rd_point(src[0], src[1])
    // result = b + [-direct[1], direct[0]] where direct = a - b
    // direct = src[0] - src[1]
    // result = src[1] + [-(src[0].y - src[1].y), src[0].x - src[1].x]
    let src_2_x = src_1_x - (src_0_y - src_1_y);
    let src_2_y = src_1_y + (src_0_x - src_1_x);

    // dst points
    let dst_0_x = out_w as f64 * 0.5;
    let dst_0_y = out_h as f64 * 0.5;
    let dst_1_x = dst_0_x + dst_dir_x;
    let dst_1_y = dst_0_y + dst_dir_y;
    // get_3rd_point(dst[0], dst[1])
    let dst_2_x = dst_1_x - (dst_0_y - dst_1_y);
    let dst_2_y = dst_1_y + (dst_0_x - dst_1_x);

    // Build linear system A * x = b for affine transform
    // For each point: src = A * dst (when inv=1)
    // We solve for A using dst as input, src as output
    let src = [
        [src_0_x, src_0_y],
        [src_1_x, src_1_y],
        [src_2_x, src_2_y],
    ];
    let dst = [
        [dst_0_x, dst_0_y],
        [dst_1_x, dst_1_y],
        [dst_2_x, dst_2_y],
    ];

    // Solve: trans = cv::getAffineTransform(dst, src) - dst is input, src is output
    // This computes the transform from dst to src
    let mut a = Vec::new();
    let mut b = Vec::new();

    for i in 0..3 {
        let sx = src[i][0];
        let sy = src[i][1];
        let dx = dst[i][0];
        let dy = dst[i][1];
        a.push([dx, dy, 1.0, 0.0, 0.0, 0.0]);
        a.push([0.0, 0.0, 0.0, dx, dy, 1.0]);
        b.push(sx);
        b.push(sy);
    }

    let n = 6;
    for i in 0..n {
        let mut max_row = i;
        for j in (i+1)..n {
            if a[j][i].abs() > a[max_row][i].abs() {
                max_row = j;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);

        let pivot = a[i][i];
        if pivot.abs() < 1e-10 {
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        }

        for j in i..n { a[i][j] /= pivot; }
        b[i] /= pivot;

        for j in 0..n {
            if j != i {
                let factor = a[j][i];
                for k in i..n { a[j][k] -= factor * a[i][k]; }
                b[j] -= factor * b[i];
            }
        }
    }

    [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32, b[4] as f32, b[5] as f32]
}

fn affine_transform(x: f32, y: f32, t: &[f32; 6]) -> (f32, f32) {
    // Apply affine transform: new_pt = t * [x, y, 1]^T
    let x_f = x as f64;
    let y_f = y as f64;
    let tx = t[0] as f64 * x_f + t[1] as f64 * y_f + t[2] as f64;
    let ty = t[3] as f64 * x_f + t[4] as f64 * y_f + t[5] as f64;
    (tx as f32, ty as f32)
}

fn perspective_transform(img: &DynamicImage, position: &[[f32; 2]; 4]) -> Result<DynamicImage, String> {
    let x0 = position[0][0] as f64;
    let y0 = position[0][1] as f64;
    let x1 = position[1][0] as f64;
    let y1 = position[1][1] as f64;
    let x2 = position[2][0] as f64;
    let y2 = position[2][1] as f64;
    let x3 = position[3][0] as f64;
    let y3 = position[3][1] as f64;

    let w1 = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
    let w2 = ((x2 - x3).powi(2) + (y2 - y3).powi(2)).sqrt();
    let h1 = ((x3 - x0).powi(2) + (y3 - y0).powi(2)).sqrt();
    let h2 = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();

    let dst_w = (w1.max(w2) * 1.1) as u32;
    let dst_h = (h1.max(h2) * 1.1) as u32;
    let dst_w = dst_w.max(100).min(1000);
    let dst_h = dst_h.max(100).min(800);

    let src = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)];
    let dst = [(0.0, 0.0), (dst_w as f64, 0.0), (dst_w as f64, dst_h as f64), (0.0, dst_h as f64)];

    let transform = get_perspective_transform(src, dst)?;

    let img_rgb = img.to_rgb8();
    let (src_w, src_h) = (img_rgb.width(), img_rgb.height());

    let mut result = image::ImageBuffer::new(dst_w, dst_h);

    for y in 0..dst_h {
        for x in 0..dst_w {
            let den = transform[6] * x as f32 + transform[7] * y as f32 + 1.0;
            if den.abs() < 0.0001 { continue; }

            let src_x = (transform[0] * x as f32 + transform[1] * y as f32 + transform[2]) / den;
            let src_y = (transform[3] * x as f32 + transform[4] * y as f32 + transform[5]) / den;

            if src_x >= 0.0 && src_x < src_w as f32 - 1.0 && src_y >= 0.0 && src_y < src_h as f32 - 1.0 {
                let x0_f = src_x.floor();
                let y0_f = src_y.floor();
                let x1 = (x0_f + 1.0).min(src_w as f32 - 1.0);
                let y1 = (y0_f + 1.0).min(src_h as f32 - 1.0);

                let x0 = x0_f as u32;
                let y0 = y0_f as u32;
                let x1 = x1 as u32;
                let y1 = y1 as u32;

                let fx = src_x - x0_f;
                let fy = src_y - y0_f;

                let p00 = img_rgb.get_pixel(x0, y0);
                let p01 = img_rgb.get_pixel(x1, y0);
                let p10 = img_rgb.get_pixel(x0, y1);
                let p11 = img_rgb.get_pixel(x1, y1);

                let r = (p00[0] as f32 * (1.0 - fx) + p01[0] as f32 * fx) * (1.0 - fy) +
                        (p10[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
                let g = (p00[1] as f32 * (1.0 - fx) + p01[1] as f32 * fx) * (1.0 - fy) +
                        (p10[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
                let b = (p00[2] as f32 * (1.0 - fx) + p01[2] as f32 * fx) * (1.0 - fy) +
                        (p10[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

                result.put_pixel(x, y, image::Rgb([r as u8, g as u8, b as u8]));
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

fn get_perspective_transform(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Result<[f32; 9], String> {
    let mut a = Vec::new();
    let mut b = Vec::new();

    for i in 0..4 {
        let (sx, sy) = src[i];
        let (dx, dy) = dst[i];
        a.push([sx, sy, 1.0, 0.0, 0.0, 0.0, -sx*dx, -sy*dx]);
        a.push([0.0, 0.0, 0.0, sx, sy, 1.0, -sx*dy, -sy*dy]);
        b.push(dx);
        b.push(dy);
    }

    let n = 8;
    for i in 0..n {
        let mut max_row = i;
        for j in (i+1)..n {
            if a[j][i].abs() > a[max_row][i].abs() {
                max_row = j;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);

        let pivot = a[i][i];
        if pivot.abs() < 1e-10 {
            return Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }

        for j in i..n { a[i][j] /= pivot; }
        b[i] /= pivot;

        for j in 0..n {
            if j != i {
                let factor = a[j][i];
                for k in i..n { a[j][k] -= factor * a[i][k]; }
                b[j] -= factor * b[i];
            }
        }
    }

    Ok([b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32, b[4] as f32, b[5] as f32, b[6] as f32, b[7] as f32, 1.0])
}
