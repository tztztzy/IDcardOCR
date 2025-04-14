#ifndef PREDICTOR_H
#define PREDICTOR_H
#include "utils.h"
#include <opencv2/dnn.hpp>


class card_correction
{
public:
	card_correction(const std::string model_path);
	myDict infer(const cv::Mat& srcimg);
private:
	const int resize_shape[2] = { 768, 768 };
	const float mean_[3] = { 0.408, 0.447, 0.470 };
	const float std_[3] = { 0.289, 0.274, 0.278 };
	const int K = 10;
	const float obj_score = 0.5;
	float c[2];
	float s;
	const int out_height = int(resize_shape[0] / 4);
	const int out_width = int(resize_shape[1] / 4);
	cv::Mat preprocess(const cv::Mat& srcimg);
	myDict postprocess(const std::vector<cv::Mat>& output, const cv::Mat& image);
	cv::Mat crop_image(const cv::Mat& img, const std::vector<cv::Point2f>& position);

	std::vector<std::string> outlayer_names;
	cv::dnn::Net model;
};

inline float distance(float x1, float y1, float x2, float y2)
{
	return sqrt(powf(x1 - x2, 2) + powf(y1 - y2, 2));
}

#endif
