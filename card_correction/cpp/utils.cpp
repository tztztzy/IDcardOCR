#include "utils.h"


using namespace std;
using namespace cv;


Mat ResizePad(const Mat& img, const int target_size, int& new_w, int& new_h, int& left, int& top)
{
	const int h = img.rows;
	const int w = img.cols;
	const int m = max(h, w);
	const float ratio = (float)target_size / (float)m;
	new_w = int(ratio * w);
	new_h = int(ratio * h);
	Mat dstimg;
	resize(img, dstimg, Size(new_w, new_h), 0, 0, INTER_LINEAR);
	top = (target_size - new_h) / 2;
	int bottom = (target_size - new_h) - top;
	left = (target_size - new_w) / 2;
	int right = (target_size - new_w) - left;
	copyMakeBorder(dstimg, dstimg, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
	return dstimg;
}

static int max_pooling(const float *input, float *output, const int inputHeight, const int inputWidth, const int outputHeight, const int outputWidth, const int kernel_h, const int kernel_w, const int outChannels, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* keep_data)
{
	int i = 0, r = 0, c = 0;
	int hstart = 0, wstart = 0, hend = 0, wend = 0;
	int h = 0, w = 0, pool_index = 0, index = 0;
	
	const float *pin = NULL;
	float *py = NULL;
	float* pkeep = NULL;
	const float flt_max = 100000;
	std::fill(output, output + outChannels * outputHeight * outputWidth, -flt_max);
	for (i = 0; i < outChannels; i++)
	{
		py = output + i * outputHeight * outputWidth;
		pin = input + i * inputHeight * inputWidth;
		pkeep = keep_data + i * outputHeight * outputWidth;
		for (r = 0; r < outputHeight; r++)
			for (c = 0; c < outputWidth; c++)
			{
				hstart = r * stride_h - pad_h;
				wstart = c * stride_w - pad_w;
				hend = std::min(hstart + kernel_h, inputHeight);
				wend = std::min(wstart + kernel_w, inputWidth);
				hstart = std::max(hstart, 0);
				wstart = std::max(wstart, 0);
				pool_index = r * outputWidth + c;
				for (h = hstart; h < hend; ++h)
					for (w = wstart; w < wend; ++w)
					{
						index = h * inputWidth + w;
						if (pin[index] > py[pool_index])
						{
							py[pool_index] = pin[index];
						}
					}
				if (py[pool_index] == pin[pool_index])
				{
					pkeep[pool_index] = 1;
				}
			}
	}
	return 0;
}

static void _nms(Mat& heat)
{
	const int kernel = 3;
	const int pad = (kernel - 1) / 2;
	const int stride = 1;
	Mat hmax = heat.clone().setTo(0.f);
	const int chan = heat.size[1];
	const int inp_h = heat.size[2];
	const int inp_w = heat.size[3];
	Mat keep = heat.clone().setTo(0.f);
	max_pooling((float*)heat.data, (float*)hmax.data, inp_h, inp_w, inp_h, inp_w, kernel, kernel, chan, pad, pad, stride, stride, (float*)keep.data);

	const vector<int> shape = { heat.size[2] , heat.size[3] };
	heat = heat.reshape(0, shape);
	keep = keep.reshape(0, shape);
	heat = heat.mul(keep);
}

static void topk_index(const float* vec, const int len, std::vector<float>& topK, std::vector<int>& topKIndex, const int topk)
{
	topK.clear();
	topKIndex.clear();
	std::vector<size_t> vec_index(len);
	std::iota(vec_index.begin(), vec_index.end(), 0);

	std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2)
	{ return vec[index_1] > vec[index_2]; });

	int k_num = std::min<int>(len, topk);
	topKIndex.resize(k_num);
	topK.resize(k_num);
	for (int i = 0; i < k_num; ++i)
	{
		const int ind = vec_index[i];
		topKIndex[i] = ind;
		topK[i] = vec[ind];
	}
}

static void _gather_feat(cv::Mat& feat, const vector<int>& ind)
{
	//cout << "feat.type():" << feat.type() << endl;
	const int dtype = feat.type();
	const int ndims = feat.size.dims();
	vector<int> newsz(ndims);
	for (int i = 0; i < ndims; i++)
	{
		newsz[i] = feat.size[i];
	}
	newsz[1] = ind.size();
	Mat new_feat;
	new_feat = Mat(newsz, CV_32FC1);
	
	for (int i = 0; i < newsz[1]; i++)
	{
		const int idx = ind[i];
		for (int j = 0; j < newsz[2]; j++)
		{
			new_feat.ptr<float>(0, i)[j] = feat.ptr<float>(0, idx)[j];
		}
	}
	new_feat.copyTo(feat);
	new_feat.release();
}


static std::tuple<Mat, vector<int>, vector<int>, Mat, Mat> _topk(const Mat& scores, const int K)
{
	const int height = scores.size[0];
	const int width = scores.size[1];
	const int len = height * width;
	
	vector<float> topk_scores;
	vector<int> topk_inds;
	topk_index((float*)scores.data, len, topk_scores, topk_inds, K);

	int num = topk_inds.size();
	vector<float> topk_ys(num);
	vector<float> topk_xs(num);
	for (int i = 0; i < num; i++)
	{
		topk_inds[i] = topk_inds[i] % len;
		topk_ys[i] = (float)topk_inds[i] / width;
		topk_xs[i] = float(topk_inds[i] % width);
	}

	vector<float> topk_score;
	vector<int> topk_ind;
	topk_index(topk_scores.data(), num, topk_score, topk_ind, K);
	num = topk_ind.size();
	vector<int> topk_clses(num);

	for (int i = 0; i < num; i++)
	{
		topk_clses[i] = int(topk_ind[i] / K);
	}

	num = int(topk_inds.size());
	const vector<int> newsz = { 1, num, 1 };
	for(int i=0;i<num;i++)
	{
		topk_inds[i] = topk_inds[topk_ind[i]];
	}
	const vector<int> out_size = { 1, K };
	Mat topk_ys_mat = Mat(newsz, CV_32FC1, topk_ys.data());

	_gather_feat(topk_ys_mat, topk_ind);
	topk_ys_mat = topk_ys_mat.reshape(0, out_size).clone();
	Mat topk_xs_mat = Mat(newsz, CV_32FC1, topk_xs.data());
	_gather_feat(topk_xs_mat, topk_ind);
	topk_xs_mat = topk_xs_mat.reshape(0, out_size).clone();

	Mat topk_score_mat = Mat(out_size, CV_32FC1, topk_score.data());

	return std::make_tuple(topk_score_mat.clone(), topk_inds, topk_clses, topk_ys_mat.clone(), topk_xs_mat.clone());
}

static void _tranpose_and_gather_feat(cv::Mat& feat, const vector<int>& ind)
{
	Mat new_feat;
	cv::transposeND(feat, { 0, 2, 3, 1 }, new_feat);
	const vector<int> newsz = { new_feat.size[0], new_feat.size[1] * new_feat.size[2], new_feat.size[3] };
	new_feat = new_feat.reshape(0, newsz);
	_gather_feat(new_feat, ind);
	new_feat.copyTo(feat);
	new_feat.release();
}

std::tuple<Mat, vector<int>> bbox_decode(cv::Mat& heat, cv::Mat& wh, cv::Mat& reg, const int K)
{
	_nms(heat);

	std::tuple<Mat, vector<int>, vector<int>, Mat, Mat> outs = _topk(heat, K);
	Mat scores = get<0>(outs).clone();
	vector<int> inds = get<1>(outs);
	vector<int> clses = get<2>(outs);
	Mat ys = get<3>(outs);
	Mat xs = get<4>(outs);

	if (!reg.empty())
	{
		_tranpose_and_gather_feat(reg, inds);
		reg = reg.reshape(0, { K,2 });
		xs = xs.reshape(0, { K, 1 }) + reg.col(0);
		ys = ys.reshape(0, { K, 1 }) + reg.col(1);

		reg = reg.reshape(0, {1, K,2 });
		xs = xs.reshape(0, { 1, K, 1 });
		ys = ys.reshape(0, { 1, K, 1 });
	}
	else
	{
		xs = xs.reshape(0, { 1, K, 1 }) + 0.5;
		ys = ys.reshape(0, { 1, K, 1 }) + 0.5;
	}
	_tranpose_and_gather_feat(wh, inds);
	wh = wh.reshape(0, { 1, K, 8 });
	scores = scores.reshape(0, { 1, K, 1 });

	const vector<int> newshape = { 1, K, wh.size[2] + scores.size[2] + 1 + xs.size[2] + ys.size[2] + 1 };
	Mat detections = Mat(newshape, CV_32FC1);
	for (int i = 0; i < K; i++)
	{
		const float x = xs.ptr<float>(0, i)[0];
		const float y = ys.ptr<float>(0, i)[0];
		detections.ptr<float>(0, i)[0] = x - wh.ptr<float>(0, i)[0];
		detections.ptr<float>(0, i)[1] = y - wh.ptr<float>(0, i)[1];
		detections.ptr<float>(0, i)[2] = x - wh.ptr<float>(0, i)[2];
		detections.ptr<float>(0, i)[3] = y - wh.ptr<float>(0, i)[3];
		detections.ptr<float>(0, i)[4] = x - wh.ptr<float>(0, i)[4];
		detections.ptr<float>(0, i)[5] = y - wh.ptr<float>(0, i)[5];
		detections.ptr<float>(0, i)[6] = x - wh.ptr<float>(0, i)[6];
		detections.ptr<float>(0, i)[7] = y - wh.ptr<float>(0, i)[7];
		detections.ptr<float>(0, i)[8] = scores.ptr<float>(0, i)[0];
		detections.ptr<float>(0, i)[9] = (float)clses[i];
		detections.ptr<float>(0, i)[10] = x;
		detections.ptr<float>(0, i)[11] = y;
	}
	
	return std::make_tuple(detections.clone(), inds);
}

Mat decode_by_ind(const cv::Mat& heat, const vector<int>& inds, const int K)
{
	Mat score = heat.clone();
	_tranpose_and_gather_feat(score, inds);
	score = score.reshape(0, { K, heat.size[1] });
	Mat Type = Mat(1, K, CV_32FC1);
	for (int i = 0; i < K; i++)
	{
		Mat row_ = score.row(i);
		double max_socre;;
		Point classIdPoint;
		cv::minMaxLoc(row_, 0, &max_socre, 0, &classIdPoint);
		Type.ptr<float>(0)[i] = (float)max_socre;
	}
	return Type;
}

static void get_dir(const float* src_point, const float rot_rad, float* src_result)
{
	float sn = sinf(rot_rad);
	float cs = cosf(rot_rad);

	src_result[0] = src_point[0] * cs - src_point[1] * sn;
	src_result[1] = src_point[0] * sn + src_point[1] * cs;
}

static void get_3rd_point(const Point2f& a, const Point2f& b, Point2f& result)
{
	Point2f direct = { a.x - b.x, a.y - b.y };
	result.x = b.x - direct.y;
	result.y = b.y + direct.x;
}

static Mat get_affine_transform(const float* center, const float scale, const int rot, const int* output_size, const int inv)
{
	const float shift[] = { 0, 0 };
	const float src_w = scale;
	const int dst_w = output_size[0];
	const int dst_h = output_size[1];
	
	float rot_rad = PI * rot / 180.f;
	float src_point[2] = { 0, src_w * -0.5f };
	float src_dir[2];
	get_dir(src_point, rot_rad, src_dir);
	float dst_dir[2] = { 0, dst_w * -0.5f };

	Point2f src[3];
	Point2f dst[3];
	src[0] = Point2f(center[0] + scale * shift[0], center[1] + scale * shift[1]);
	src[1] = Point2f(center[0] + src_dir[0] + scale * shift[0], center[1] + src_dir[1] + scale * shift[1]);
	dst[0] = Point2f(dst_w * 0.5f, dst_h * 0.5f);
	dst[1] = Point2f(dst_w * 0.5f + dst_dir[0], dst_h * 0.5f + dst_dir[1]);

	get_3rd_point(src[0], src[1], src[2]);
	get_3rd_point(dst[0], dst[1], dst[2]);
	Mat trans;
	if (inv == 1)
	{
		trans = cv::getAffineTransform(dst, src);
	}
	else
	{
		trans = cv::getAffineTransform(src, dst);
	}
	return trans;
}

static void affine_transform(float* pt, const cv::Mat& t)
{
	Mat new_pt = (Mat_<double>(3, 1) << pt[0], pt[1], 1.0);
	Mat tmp = t * new_pt;
	pt[0] = (float)tmp.ptr<double>(0)[0];
	pt[1] = (float)tmp.ptr<double>(1)[0];
}

void bbox_post_process(cv::Mat& bbox, const float* c, const float s, const int h, const int w)
{
	const int num = bbox.size[1];
	const int len = bbox.size[2];
	vector<int> newshape = { num, len };
	bbox = bbox.reshape(0, newshape);
	const int output_size[2] = { w,h };
	Mat trans = get_affine_transform(c, s, 0, output_size, 1);
	float* pdata = (float*)bbox.data;
	for(int i=0;i<num;i++)
	{
		for(int j=0;j<8;j+=2)
		{
			affine_transform(pdata+j, trans);
		}
		affine_transform(pdata+10, trans);
		pdata += len;
	}
}

void draw_show_img(Mat img, myDict result, string savepath)
{
	vector<vector<Point2f>> polys = std::get<vector<vector<Point2f>>>(result["POLYGONS"]);
	vector<Point2f> centers = std::get<vector<Point2f>>(result["CENTER"]);
	vector<int> angle_cls = std::get<vector<int>>(result["LABELS"]);
	vector<vector<int>> bbox = std::get<vector<vector<int>>>(result["BBOX"]);
	Scalar color = Scalar(0, 0, 255);
	for(int i=0;i<polys.size();i++)
	{
		vector<vector<Point>> cnts(1);
		for(int j=0;j<polys[i].size();j++)
		{
			cnts[0].emplace_back(Point(int(polys[i][j].x), int(polys[i][j].y)));
		}
		
		Point ori_center = Point((bbox[i][0]+bbox[i][2])/2, (bbox[i][1]+bbox[i][3])/2);
		cv::drawContours(img, cnts, -1, color, 2);    ////Point2f的运行时会报错
		cv::circle(img, Point(int(centers[i].x), int(centers[i].y)), 5, color, 2);
		cv::circle(img, ori_center, 5, color, 2);
		cv::putText(img, to_string(angle_cls[i]), ori_center, FONT_HERSHEY_SIMPLEX, 2, color, 2);
	}
	cv::imwrite(savepath, img);
}

void merge_images_horizontal(vector<Mat> images, string output_path)
{
	int target_height = images[0].rows;
	for(int i=1;i<images.size();i++)
	{
		target_height = std::min(target_height, images[i].rows);
	}
	const int num_img = images.size();
	vector<Mat> resized_images(num_img);
	int total_width = 0;
	for(int i=0;i<num_img;i++)
	{
		float aspect_ratio = (float)images[i].cols / (float)images[i].rows;
		int new_width = int(target_height * aspect_ratio);
		cv::resize(images[i], resized_images[i], Size(new_width, target_height));
		total_width += resized_images[i].cols;
	}

	Mat merged_image = Mat(target_height, total_width, CV_8UC3, Scalar(0, 0, 0));

	int x_offset = 0;
	for(int i=0;i<num_img;i++)
	{
		resized_images[i].copyTo(merged_image.colRange(x_offset, x_offset + resized_images[i].cols));
		x_offset += resized_images[i].cols;
	}

	cv::imwrite(output_path, merged_image);
}