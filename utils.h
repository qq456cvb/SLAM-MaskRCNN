#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <map>

cv::Mat parse_pos(const std::vector<double>& list);
cv::Mat inv_extrinsic(const cv::Mat& extrinsic);
cv::Mat mult_extrinsic(const cv::Mat& extrinsic1, const cv::Mat& extrinsic2);
cv::Mat pack_tsdf_color(const cv::Mat& tsdf, const cv::Mat& color);
std::map<double, std::vector<double>> read_trajactory(std::string filename);

