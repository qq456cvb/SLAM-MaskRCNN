#pragma once
#include <opencv2/opencv.hpp>
class TSDF
{
public:
	TSDF(cv::Scalar intrinsics);
	float mean_depth_;
	void parse_frame(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsic, float);
	uint8_t *get_tsdf_color() const;
	float *get_tsdf_diff() const;
	cv::Vec3i get_dim() const;
	cv::Vec3f get_vol_start() const;
	cv::Vec3f get_vol_end() const;
	cv::Vec3f get_voxel() const;
	cv::Mat get_intrinsic() const;
	cv::Mat get_intrinsic_inv() const;
	~TSDF();

private:
	float *tsdf_diff_;
	int *tsdf_wt_;
	uint8_t *tsdf_color_;
	cv::Scalar  miu_;
	cv::Vec3i vol_dim_ = cv::Vec3i(256, 256, 256);
	cv::Vec3f vol_res_;
	cv::Vec3f vol_start_, vol_end_;
	cv::Mat intrinsic_ = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat intrinsic_inv_;
	cv::Mat init_extrinsic_inv_ = cv::Mat(4, 4, CV_32F);
	bool init_ = false;
	void launch_kernel(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsic);
};

