#pragma once
#include <opencv2/opencv.hpp>
class TSDF
{
public:
	TSDF(cv::Scalar intrinsics);
	void parse_frame(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsics, double);
	cv::Mat get_tsdf_color() const;
	cv::Mat get_tsdf() const;
	cv::Vec3i get_dim() const;
	cv::Vec3d get_vol_start() const;
	cv::Vec3d get_vol_end() const;
	cv::Mat get_intrinsics() const;
	~TSDF();

private:
	cv::Mat tsdf_;
	cv::Mat tsdf_wt_;
	cv::Mat tsdf_color_;
	cv::Scalar  mu_;
	cv::Vec3i vol_dim_ = cv::Vec3i(64, 64, 64);
	cv::Vec2i tex_dim_ = cv::Vec2i(sqrt(vol_dim_[0] * vol_dim_[1] * vol_dim_[2]),
		sqrt(vol_dim_[0] * vol_dim_[1] * vol_dim_[2]));
	cv::Vec3d vol_res_;
	cv::Vec3d vol_start_, vol_end_;
	cv::Mat intrinsics_ = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat intrinsics_inv_;
	cv::Mat init_pos_inv_ = cv::Mat(3, 4, CV_64F);
	bool init_ = false;
};

