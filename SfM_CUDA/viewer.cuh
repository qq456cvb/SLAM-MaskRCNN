#pragma once
#include "tsdf.cuh"

class Viewer {
public:
	float *s2w_d;
	float *c_d;
	uchar3 *output_d;
	int width_;
	int height_;

	explicit Viewer(int width, int height);
	~Viewer();

	cv::Mat show_tsdf(const TSDF& tsdf, float angle, float dist);
};
