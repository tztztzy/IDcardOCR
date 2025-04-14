#include "predictor.h"


using namespace std;
using namespace cv;
using namespace dnn;


card_correction::card_correction(const string model_path)
{
	this->model = readNet(model_path);
	this->outlayer_names = this->model.getUnconnectedOutLayersNames();
}

Mat card_correction::preprocess(const Mat& srcimg)
{
	Mat img;
	int new_w, new_h, left, top;
	img = ResizePad(srcimg, this->resize_shape[0], new_w, new_h, left, top);

	vector<cv::Mat> bgrChannels(3);
	split(img, bgrChannels);
	for (int c = 0; c < 3; c++)
	{
		bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1.0 / (255.0* std_[c]), (0.0 - mean_[c]) / std_[c]);
	}
	Mat m_normalized_mat;
	merge(bgrChannels, m_normalized_mat);

	Mat blob = blobFromImage(m_normalized_mat);
	return blob;
}

myDict card_correction::infer(const Mat& srcimg)
{
	const int ori_h = srcimg.rows;
	const int ori_w = srcimg.cols;
	this->c[0] = (float)ori_w / 2.f;
	this->c[1] = (float)ori_h / 2.f;
	this->s = std::max(ori_h, ori_w) * 1.f;
	Mat blob = this->preprocess(srcimg);

	this->model.setInput(blob);
	std::vector<Mat> pre_out;
	this->model.forward(pre_out, this->outlayer_names);

	myDict out = this->postprocess(pre_out, srcimg);
	return out;
}


static Mat sigmoid(Mat x)
{
	Mat y;
	cv::exp(-x, y);
	y = 1.f / (1 + y);
	return y;
}

Mat card_correction::crop_image(const Mat& img, const vector<Point2f>& position)
{
	float img_width = distance((position[0].x + position[3].x)*0.5, (position[0].y + position[3].y)*0.5, (position[1].x + position[2].x)*0.5, (position[1].y + position[2].y)*0.5);
	float img_height = distance((position[0].x + position[1].x)*0.5, (position[0].y + position[1].y)*0.5, (position[2].x + position[3].x)*0.5, (position[2].y + position[3].y)*0.5);

	vector<Point2f> corners_trans = { Point2f(0,0), Point2f(img_width, 0), Point2f(img_width, img_height), Point2f(0, img_height) };

	Mat transform = cv::getPerspectiveTransform(position, corners_trans);
	Mat dst;
	cv::warpPerspective(img, dst, transform, Size(int(img_width), int(img_height)));
	return dst;
}

myDict card_correction::postprocess(const std::vector<cv::Mat>& output, const cv::Mat& image)
{
	Mat reg = output[3];  ////shape: (1, 2, 192, 192)
	Mat wh = output[2];        ////shape: (1, 8, 192, 192)
	Mat hm = sigmoid(output[4]);     ////shape: (1, 1, 192, 192)
	Mat angle_cls = sigmoid(output[0]);   ////shape: (1, 4, 192, 192)
	Mat ftype_cls = sigmoid(output[1]);   ////shape: (1, 2, 192, 192)

	std::tuple<Mat, vector<int>> outs = bbox_decode(hm, wh, reg, this->K);
	Mat bbox = get<0>(outs);
	vector<int> inds = get<1>(outs);
	angle_cls = decode_by_ind(angle_cls, inds, this->K);
	ftype_cls = decode_by_ind(ftype_cls, inds, this->K);

	for (int i = 0; i < bbox.size[1]; i++)
	{
		bbox.ptr<float>(0, i)[9] = angle_cls.ptr<float>(0)[i];
		bbox.ptr<float>(0, i)[12] = ftype_cls.ptr<float>(0)[i];
	}

	bbox_post_process(bbox, this->c, this->s, this->out_height, this->out_width);

	vector<vector<Point2f>> res;
	vector<int> angle;
	vector<Mat> sub_imgs;
	vector<vector<int>> corner_left_right;
	vector<int> ftype;
	vector<float> score;
	vector<Point2f> center;
	for (int i = 0; i < bbox.size[0]; i++)
	{
		if (bbox.ptr<float>(i)[8] > this->obj_score)
		{
			const int angle_data = int(bbox.ptr<float>(i)[9]);
			angle.emplace_back(angle_data);
			vector<Point2f> box8point(4);
			int min_x = 10000, min_y = 10000;
			int max_x = -10000, max_y = -10000;
			for (int j = 0; j < 4; j += 1)
			{
				const float x = bbox.ptr<float>(i)[2 * j];
				const float y = bbox.ptr<float>(i)[2 * j + 1];
				box8point[j] = { x,y };
				min_x = std::min((int)x, min_x);
				min_y = std::min((int)y, min_y);
				max_x = std::max((int)x, max_x);
				max_y = std::max((int)y, max_y);
			}
			vector<int> corner_left_right_data = { min_x, min_y, max_x, max_y };
			corner_left_right.emplace_back(corner_left_right_data);
			res.emplace_back(box8point);
			Mat sub_img = this->crop_image(image, box8point);
			if (angle_data == 1)
			{
				cv::rotate(sub_img, sub_img, 2);
			}
			if (angle_data == 2)
			{
				cv::rotate(sub_img, sub_img, 1);
			}
			if (angle_data == 3)
			{
				cv::rotate(sub_img, sub_img, 0);
			}
			sub_imgs.emplace_back(sub_img);
			ftype.emplace_back(int(bbox.ptr<float>(i)[12]));
			score.emplace_back(bbox.ptr<float>(i)[8]);
			center.emplace_back(Point2f(bbox.ptr<float>(i)[10], bbox.ptr<float>(i)[11]));
		}
	}
	myDict result;
	result["POLYGONS"] = res;
	result["BBOX"] = corner_left_right;
	result["SCORES"] = score;
	result["OUTPUT_IMGS"] = sub_imgs;
	result["LABELS"] = angle;
	result["LAYOUT"] = ftype;
	result["CENTER"] = center;
	return result;
}