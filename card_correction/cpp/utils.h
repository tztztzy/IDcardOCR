#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <numeric>
#include <vector>
#include <map>
#include <variant>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#define PI 3.14159265358979323846

////std::variant是c++17里的特性
typedef std::map<std::string, std::variant<std::vector<std::vector<cv::Point2f>>, std::vector<float>, std::vector<int>, std::vector<std::vector<int>>, std::vector<cv::Mat>, std::vector<cv::Point2f>>> myDict;

cv::Mat ResizePad(const cv::Mat& img, const int target_size, int& new_w, int& new_h, int& left, int& top);

std::tuple<cv::Mat, std::vector<int>> bbox_decode(cv::Mat& heat, cv::Mat& wh, cv::Mat& reg, const int K=100);
cv::Mat decode_by_ind(const cv::Mat& heat, const std::vector<int>& inds, const int K = 100);
void bbox_post_process(cv::Mat& bbox, const float* c, const float s, const int h, const int w);

void draw_show_img(cv::Mat img, myDict result, std::string savepath);
void merge_images_horizontal(std::vector<cv::Mat> images, std::string output_path);

#endif