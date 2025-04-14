#ifndef CARD_CORRECTION_DLL_H
#define CARD_CORRECTION_DLL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef CARD_CORRECTION_EXPORTS
        #define CARD_API __declspec(dllexport)
    #else
        #define CARD_API __declspec(dllimport)
    #endif
#else
    #define CARD_API __attribute__((visibility("default")))
#endif

// 错误码定义
#define CARD_SUCCESS 0
#define CARD_ERROR_INVALID_HANDLE -1
#define CARD_ERROR_MODEL_LOAD -2
#define CARD_ERROR_INFERENCE -3
#define CARD_ERROR_INVALID_INPUT -4
#define CARD_ERROR_MEMORY -5

// 检测结果结构体
typedef struct {
    float x0, y0;  // 左上角
    float x1, y1;  // 右上角
    float x2, y2;  // 右下角
    float x3, y3;  // 左下角
    float score;   // 置信度
    int angle;     // 旋转角度 (0-3)
    float cx, cy;  // 中心点
    int ftype;     // 类型
} CardDetection;

// 矫正结果结构体
typedef struct {
    CardDetection* detections;  // 检测结果数组
    int num_detections;         // 检测数量
    unsigned char** output_images;  // 输出的矫正后图像数据（JPEG格式）
    int* output_image_sizes;    // 每张图像的大小（字节）
    int* output_image_angles;   // 每张图像的旋转角度
    int num_output_images;      // 输出图像数量
} CardCorrectionResult;

// Opaque handle for the correction model
typedef void* CardCorrectionHandle;

// 创建模型实例
// model_path: ONNX模型文件路径
// 返回: 模型句柄，失败返回NULL
CARD_API CardCorrectionHandle card_correction_create(const char* model_path);

// 释放模型实例
CARD_API void card_correction_destroy(CardCorrectionHandle handle);

// 执行身份证矫正
// handle: 模型句柄
// image_data: 输入图像数据（BGR格式，连续内存）
// width: 图像宽度
// height: 图像高度
// channels: 通道数（通常为3）
// result: 输出结果（调用者需要在使用后调用 card_correction_free_result 释放）
// 返回: 错误码
CARD_API int card_correction_infer(
    CardCorrectionHandle handle,
    const unsigned char* image_data,
    int width,
    int height,
    int channels,
    CardCorrectionResult** result
);

// 释放推理结果
CARD_API void card_correction_free_result(CardCorrectionResult* result);

// 获取错误信息
CARD_API const char* card_correction_get_error_string(int error_code);

// 获取版本信息
CARD_API const char* card_correction_get_version();

#ifdef __cplusplus
}
#endif

#endif // CARD_CORRECTION_DLL_H
