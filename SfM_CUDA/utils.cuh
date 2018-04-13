#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <map>
#include <cuda_runtime.h>

cv::Mat parse_extrinsic(const std::vector<double>& list);
cv::Mat inv_extrinsic(const cv::Mat& extrinsic);
cv::Mat mult_extrinsic(const cv::Mat& extrinsic1, const cv::Mat& extrinsic2);
cv::Mat pack_tsdf_color(float* tsdf, uint8_t* color);
float mean_depth(const cv::Mat& depth);
std::map<double, std::vector<double>> read_trajactory(std::string filename);

__device__ float interp_tsdf_diff(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, float *tsdf_diff);
__device__ uchar3 interp_tsdf_color(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, uchar3 *tsdf_color);
__device__ void interp_tsdf_cnt(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, uint32_t *tsdf_cnt, float *out);


