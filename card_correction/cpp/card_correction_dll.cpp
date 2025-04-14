#include "card_correction_dll.h"
#include "predictor.h"
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <cstring>
#include <sstream>

// 内部使用的上下文结构
typedef struct {
    card_correction* model;
    std::string last_error;
} CardCorrectionContext;

static const char* VERSION = "1.0.0";

// 错误信息映射
static const char* get_error_message(int error_code) {
    switch (error_code) {
        case CARD_SUCCESS: return "Success";
        case CARD_ERROR_INVALID_HANDLE: return "Invalid handle";
        case CARD_ERROR_MODEL_LOAD: return "Failed to load model";
        case CARD_ERROR_INFERENCE: return "Inference error";
        case CARD_ERROR_INVALID_INPUT: return "Invalid input";
        case CARD_ERROR_MEMORY: return "Memory allocation error";
        default: return "Unknown error";
    }
}

CARD_API CardCorrectionHandle card_correction_create(const char* model_path) {
    if (!model_path) {
        return nullptr;
    }
    
    try {
        CardCorrectionContext* ctx = new CardCorrectionContext();
        ctx->model = new card_correction(std::string(model_path));
        return reinterpret_cast<CardCorrectionHandle>(ctx);
    } catch (const std::exception& e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

CARD_API void card_correction_destroy(CardCorrectionHandle handle) {
    if (!handle) return;
    
    CardCorrectionContext* ctx = reinterpret_cast<CardCorrectionContext*>(handle);
    if (ctx->model) {
        delete ctx->model;
    }
    delete ctx;
}

CARD_API int card_correction_infer(
    CardCorrectionHandle handle,
    const unsigned char* image_data,
    int width,
    int height,
    int channels,
    CardCorrectionResult** result
) {
    if (!handle) {
        return CARD_ERROR_INVALID_HANDLE;
    }
    
    if (!image_data || !result || width <= 0 || height <= 0 || channels <= 0) {
        return CARD_ERROR_INVALID_INPUT;
    }
    
    CardCorrectionContext* ctx = reinterpret_cast<CardCorrectionContext*>(handle);
    
    try {
        // 创建OpenCV Mat (注意：OpenCV使用BGR格式)
        cv::Mat image(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, 
                     const_cast<unsigned char*>(image_data));
        
        // 执行推理
        myDict output = ctx->model->infer(image);
        
        // 提取结果
        auto polygons = std::get<std::vector<std::vector<cv::Point2f>>>(output["POLYGONS"]);
        auto bbox = std::get<std::vector<std::vector<int>>>(output["BBOX"]);
        auto scores = std::get<std::vector<float>>(output["SCORES"]);
        auto output_imgs = std::get<std::vector<cv::Mat>>(output["OUTPUT_IMGS"]);
        auto labels = std::get<std::vector<int>>(output["LABELS"]);
        auto layout = std::get<std::vector<int>>(output["LAYOUT"]);
        auto center = std::get<std::vector<cv::Point2f>>(output["CENTER"]);
        
        int num_detections = static_cast<int>(polygons.size());
        
        // 分配结果结构
        CardCorrectionResult* res = new CardCorrectionResult();
        std::memset(res, 0, sizeof(CardCorrectionResult));
        
        res->num_detections = num_detections;
        res->num_output_images = static_cast<int>(output_imgs.size());
        
        if (num_detections > 0) {
            // 分配检测数组
            res->detections = new CardDetection[num_detections];
            
            for (int i = 0; i < num_detections && i < static_cast<int>(polygons.size()); i++) {
                const auto& poly = polygons[i];
                if (poly.size() >= 4) {
                    res->detections[i].x0 = poly[0].x;
                    res->detections[i].y0 = poly[0].y;
                    res->detections[i].x1 = poly[1].x;
                    res->detections[i].y1 = poly[1].y;
                    res->detections[i].x2 = poly[2].x;
                    res->detections[i].y2 = poly[2].y;
                    res->detections[i].x3 = poly[3].x;
                    res->detections[i].y3 = poly[3].y;
                }
                
                if (i < static_cast<int>(scores.size())) {
                    res->detections[i].score = scores[i];
                }
                
                if (i < static_cast<int>(labels.size())) {
                    res->detections[i].angle = labels[i];
                }
                
                if (i < static_cast<int>(center.size())) {
                    res->detections[i].cx = center[i].x;
                    res->detections[i].cy = center[i].y;
                }
                
                if (i < static_cast<int>(layout.size())) {
                    res->detections[i].ftype = layout[i];
                }
            }
        }
        
        // 处理输出图像（编码为JPEG）
        int num_images = static_cast<int>(output_imgs.size());
        if (num_images > 0) {
            res->output_images = new unsigned char*[num_images];
            res->output_image_sizes = new int[num_images];
            res->output_image_angles = new int[num_images];
            
            for (int i = 0; i < num_images; i++) {
                std::vector<uchar> buf;
                cv::imencode(".jpg", output_imgs[i], buf);
                
                res->output_image_sizes[i] = static_cast<int>(buf.size());
                res->output_images[i] = new unsigned char[buf.size()];
                std::memcpy(res->output_images[i], buf.data(), buf.size());
                
                if (i < static_cast<int>(labels.size())) {
                    res->output_image_angles[i] = labels[i];
                } else {
                    res->output_image_angles[i] = 0;
                }
            }
        }
        
        *result = res;
        return CARD_SUCCESS;
        
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return CARD_ERROR_INFERENCE;
    } catch (...) {
        ctx->last_error = "Unknown error during inference";
        return CARD_ERROR_INFERENCE;
    }
}

CARD_API void card_correction_free_result(CardCorrectionResult* result) {
    if (!result) return;
    
    // 释放检测结果数组
    if (result->detections) {
        delete[] result->detections;
    }
    
    // 释放输出图像
    if (result->output_images) {
        for (int i = 0; i < result->num_output_images; i++) {
            if (result->output_images[i]) {
                delete[] result->output_images[i];
            }
        }
        delete[] result->output_images;
    }
    
    // 释放图像大小数组
    if (result->output_image_sizes) {
        delete[] result->output_image_sizes;
    }
    
    // 释放图像角度数组
    if (result->output_image_angles) {
        delete[] result->output_image_angles;
    }
    
    delete result;
}

CARD_API const char* card_correction_get_error_string(int error_code) {
    return get_error_message(error_code);
}

CARD_API const char* card_correction_get_version() {
    return VERSION;
}
