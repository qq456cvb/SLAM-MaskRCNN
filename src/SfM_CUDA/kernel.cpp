#include <iostream>
#include "tsdf.cuh"
#include "utils.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <memory>
#include "viewer.cuh"


using namespace std;


string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main()
{
	cv::Scalar intrinsics{ 520.9f, 521.0f, 325.1f, 249.7f };
	auto tsdf = std::make_shared<TSDF>(intrinsics);
	cv::String rgb_path("D://rgb-datasets//desk//rgb//*.png"); //select only png
	cv::String depth_path("D://rgb-datasets//desk//depth//*.png");
	cv::String mask_path("D://rgb-datasets//desk//mask//*.png");
	auto traj = read_trajactory("D://rgb-datasets//desk//groundtruth.txt");
	vector<cv::String> rgb_fn, depth_fn, mask_fn;
	cv::glob(rgb_path, rgb_fn, false);
	cv::glob(depth_path, depth_fn, false);
	cv::glob(mask_path, mask_fn, false);
	size_t j = 0;
	vector<double> depth_timestamps, mask_timestamps;
	transform(depth_fn.begin(), depth_fn.end(), back_inserter(depth_timestamps),
		[](const cv::String& fn) -> float {
		return stod(fn.substr(fn.find_last_of("/") + 6, fn.find_last_of(".") - fn.find_last_of("/") - 6));
	});
	transform(mask_fn.begin(), mask_fn.end(), back_inserter(mask_timestamps),
		[](const cv::String& fn) -> float {
		return stod(fn.substr(fn.find_last_of("/") + 6, fn.find_last_of(".") - fn.find_last_of("/") - 6));
	});

	double begin = 68164;
	double end = 68170;
	Viewer *viewer = nullptr;
	int cnt = 0;
	for (size_t i = 0; i < 10000; i++) {
		if (i >= depth_timestamps.size()) break;
		if (depth_timestamps[i] < begin || depth_timestamps[i] > end) continue;
		while (depth_timestamps[i] < mask_timestamps[j]) i++;
		while (mask_timestamps[j] < depth_timestamps[i]) j++;
		auto depth_img = cv::imread(depth_fn[i], CV_LOAD_IMAGE_ANYDEPTH);
		auto mask_img = cv::imread(mask_fn[j], CV_LOAD_IMAGE_GRAYSCALE);
		auto rgb_img = cv::imread(rgb_fn[j]);
		std::cout << "processing: " << i << ", " << rgb_fn[j] << std::endl;
		cnt++;
		if (cnt > 100) break;
		//if (cnt > 1 && cnt < 99) continue;
		if (!viewer)
		{
			viewer = new Viewer(depth_img.cols, depth_img.rows);
		}
		//cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);

		// TODO: there are small noisy objects in mrcnn...
		cv::Mat obj_img(mask_img.rows, mask_img.cols, CV_8UC1, cv::Scalar(0));
		auto mask_ptr = mask_img.data;
		for (int k = 0; k < mask_img.rows * mask_img.cols; k++)
		{
			if (mask_ptr[k] > 0)
			{
				obj_img.data[k] = mask_ptr[k] * 20;
			}
		}
		cv::imshow("mask", obj_img);
		cv::waitKey(30);

		auto mean = mean_depth(depth_img);

		auto low = traj.lower_bound(depth_timestamps[i]);
		auto extrinsic = parse_extrinsic(low->second);
		tsdf->parse_frame(depth_img, rgb_img, mask_img, extrinsic, mean);
	}
	if (viewer) {
		float angle = 0.f;
		while (1) {
			angle += 0.01f;
			viewer->show_tsdf(*tsdf, angle, tsdf->mean_depth_);
		}
	}
	
	delete viewer;
	return 0;
}
