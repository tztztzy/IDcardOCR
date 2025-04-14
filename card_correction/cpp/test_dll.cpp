
#include <iostream>
#include <cstring>
#include "card_correction_dll.h"
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    cout << "========================================" << endl;
    cout << "  身份证矫正DLL测试程序" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // 获取版本信息
    cout << "DLL版本: " << card_correction_get_version() << endl;
    cout << endl;
    
    // 参数检查
    if (argc < 3) {
        cout << "用法: " << argv[0] << " <模型路径> <图片路径>" << endl;
        cout << "示例: " << argv[0] << " ..\\..\\models\\card_correction.onnx ..\\testimgs\\demo.jpg" << endl;
        return -1;
    }
    
    string model_path = argv[1];
    string img_path = argv[2];
    
    cout << "模型路径: " << model_path << endl;
    cout << "图片路径: " << img_path << endl;
    cout << endl;
    
    // 创建模型实例
    cout << "正在加载模型..." << endl;
    CardCorrectionHandle handle = card_correction_create(model_path.c_str());
    if (!handle) {
        cerr << "错误: 无法创建模型实例" << endl;
        return -1;
    }
    cout << "模型加载成功!" << endl;
    cout << endl;
    
    // 读取图像
    cout << "正在读取图像..." << endl;
    Mat img = imread(img_path);
    if (img.empty()) {
        cerr << "错误: 无法读取图像: " << img_path << endl;
        card_correction_destroy(handle);
        return -1;
    }
    cout << "图像尺寸: " << img.cols << "x" << img.rows << endl;
    cout << endl;
    
    // 执行推理
    cout << "正在执行推理..." << endl;
    CardCorrectionResult* result = nullptr;
    int ret = card_correction_infer(
        handle,
        img.data,
        img.cols,
        img.rows,
        img.channels(),
        &result
    );
    
    if (ret != CARD_SUCCESS) {
        cerr << "错误: 推理失败，错误码: " << ret << endl;
        cerr << "错误信息: " << card_correction_get_error_string(ret) << endl;
        card_correction_destroy(handle);
        return -1;
    }
    cout << "推理完成!" << endl;
    cout << endl;
    
    // 显示结果
    cout << "========================================" << endl;
    cout << "  推理结果" << endl;
    cout << "========================================" << endl;
    cout << "检测到 " << result->num_detections << " 个目标" << endl;
    cout << "输出 " << result->num_output_images << " 张图像" << endl;
    cout << endl;
    
    for (int i = 0; i < result->num_detections; i++) {
        CardDetection& det = result->detections[i];
        cout << "检测 " << (i + 1) << ":" << endl;
        cout << "  置信度: " << det.score << endl;
        cout << "  角度: " << det.angle << endl;
        cout << "  类型: " << det.ftype << endl;
        cout << "  中心点: (" << det.cx << ", " << det.cy << ")" << endl;
        cout << "  四边形坐标:" << endl;
        cout << "    (" << det.x0 << ", " << det.y0 << ")" << endl;
        cout << "    (" << det.x1 << ", " << det.y1 << ")" << endl;
        cout << "    (" << det.x2 << ", " << det.y2 << ")" << endl;
        cout << "    (" << det.x3 << ", " << det.y3 << ")" << endl;
        cout << endl;
    }
    
    // 保存输出图像
    for (int i = 0; i < result->num_output_images; i++) {
        string output_path = "output_" + to_string(i) + ".jpg";
        FILE* fp = fopen(output_path.c_str(), "wb");
        if (fp) {
            fwrite(result->output_images[i], 1, result->output_image_sizes[i], fp);
            fclose(fp);
            cout << "已保存: " << output_path << " (" << result->output_image_sizes[i] << " 字节)" << endl;
        }
    }
    
    // 释放资源
    card_correction_free_result(result);
    card_correction_destroy(handle);
    
    cout << endl;
    cout << "测试完成!" << endl;
    
    return 0;
}
