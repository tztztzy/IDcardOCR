

#include "predictor.h"

using namespace std;
using namespace cv;

int main()
{
	string imgpath = "./testimgs/demo3.jpg";    ////图片路径要写正确
	string modelpath = "./card_correction.onnx";

	card_correction mynet(modelpath);
	Mat srcimg = imread(imgpath);
	myDict out = mynet.infer(srcimg);

	draw_show_img(srcimg.clone(), out, "show.jpg");
	vector<Mat> sub_imgs = std::get<vector<Mat>>(out["OUTPUT_IMGS"]);
	sub_imgs.insert(sub_imgs.begin(), srcimg);
	merge_images_horizontal(sub_imgs, "pp4_rotate_show.jpg");

	return 0;
}
