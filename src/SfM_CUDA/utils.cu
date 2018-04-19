#include "utils.cuh"
#include <sstream>
#include <fstream>
#include "helper_math.h"
#include "tsdf.cuh"

// parse camera position to projection matrix
cv::Mat parse_extrinsic(const std::vector<double>& list) {
	cv::Vec3d axis{ list[3], list[4], list[5] };
	double axis_norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
	double theta = 2 * atan2(axis_norm, list[6]);
	axis = axis / axis_norm;

	cv::Mat rotation;
	cv::Vec3d rod = theta * axis;
	cv::Rodrigues(rod, rotation);

	cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);
	rotation.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));
	cv::Mat translation(3, 1, CV_64F, (void*)list.data());
	translation.copyTo(extrinsic(cv::Rect(3, 0, 1, 3)));
	extrinsic.convertTo(extrinsic, CV_32F);
	return extrinsic.inv();
}

cv::Mat inv_extrinsic(const cv::Mat& extrinsic) {
	cv::Mat rotation = extrinsic(cv::Rect(0, 0, 3, 3));
	cv::Mat translation = extrinsic(cv::Rect(3, 0, 1, 3));
	cv::Mat result(3, 4, CV_64F);
	cv::Mat rotation_inv = rotation.inv();
	cv::Mat t_prime = -rotation_inv * translation;
	rotation_inv.copyTo(result(cv::Rect(0, 0, 3, 3)));
	t_prime.copyTo(result(cv::Rect(3, 0, 1, 3)));
	std::cout << result << std::endl;
	return result;
}

cv::Mat mult_extrinsic(const cv::Mat& extrinsic1, const cv::Mat& extrinsic2) {
	cv::Mat result(3, 4, CV_64F);
	result(cv::Rect(0, 0, 3, 3)) = extrinsic1(cv::Rect(0, 0, 3, 3)) * extrinsic2(cv::Rect(0, 0, 3, 3));
	result(cv::Rect(3, 0, 1, 3)) = extrinsic1(cv::Rect(0, 0, 3, 3)) * extrinsic2(cv::Rect(3, 0, 1, 3)) + extrinsic1(cv::Rect(3, 0, 1, 3));
	return result;
}

cv::Mat pack_tsdf_color(float* tsdf_ptr, uint8_t* color_ptr) {
	cv::Mat color(4096, 4096, CV_8UC3, color_ptr);
	cv::Mat tsdf(4096, 4096, CV_32FC1, tsdf_ptr);
	cv::Mat result(color.rows, color.cols, CV_32FC4, cv::Scalar(0));
	cv::Mat color_normed;
	color.convertTo(color_normed, CV_32FC3, 1. / 255.);
	cv::Mat colors[4];
	cv::split(color_normed, colors);
	colors[3] = tsdf;
	cv::merge(colors, 4, result);
	/*cv::Mat test[4];
	cv::split(result, test);*/
	//cv::imshow("test", color_normed);
	//cv::waitKey(0);
	return result;
}

std::map<double, std::vector<double> > read_trajactory(std::string filename) {
	std::map<double, std::vector<double> > result;
	std::string line;
	std::ifstream infile(filename.c_str());
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		double ts, tx, ty, tz, qx, qy, qz, qw;
		if (!(iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
		std::vector<double> pos = { tx, ty, tz, qx, qy, qz, qw };
		result.insert(std::make_pair(std::fmod(ts, 1e5), pos));
	}
	return result;
}

float mean_depth(const cv::Mat& depth) {
	int cnt = depth.rows * depth.cols;
	uint16_t *ptr = (uint16_t*)depth.data;
	double sum = 0;
	int total = 0;
	for (int i = 0; i < cnt; i++) {
		if (ptr[i] == 0)
		{
			continue;
		}
		sum += ptr[i] / 5000.;
		total++;
	}
	return static_cast<float>(sum / total);
}

template <typename T>
__device__ T mix(T a, T b, float interp) {
	return (1 - interp) * a + interp * b;
}


__device__ float interp_tsdf_diff(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, float *tsdf_diff) {
	float3 idx = (pos - vol_start) / voxel;
	int3 floored_idx = make_int3(floorf(idx.x), floorf(idx.y), floorf(idx.z));
	float3 frac_idx = idx - make_float3(floored_idx.x, floored_idx.y, floored_idx.z);
	int base_idx = vol_dim.y * vol_dim.z * floored_idx.x + vol_dim.z * floored_idx.y + floored_idx.z;
	float diffs[8];
	for (uint8_t i = 0; i < 2; ++i)
	{
		for (uint8_t j = 0; j < 2; ++j)
		{
			for (uint8_t k = 0; k < 2; ++k)
			{
				int vol_idx = base_idx + vol_dim.y * vol_dim.z * i + vol_dim.z * j + k;
				diffs[i * 4 + j * 2 + k] = tsdf_diff[vol_idx];
			}
		}
	}
	float low = mix(mix(diffs[0], diffs[4], frac_idx.x), mix(diffs[2], diffs[6], frac_idx.x), frac_idx.y);
	float high = mix(mix(diffs[1], diffs[5], frac_idx.x), mix(diffs[3], diffs[7], frac_idx.x), frac_idx.y);
	return mix(low, high, frac_idx.z);
}

__device__ uchar3 interp_tsdf_color(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, uchar3 *tsdf_color) {
	float3 idx = (pos - vol_start) / voxel;
	int3 floored_idx = make_int3(floorf(idx.x), floorf(idx.y), floorf(idx.z));
	float3 frac_idx = idx - make_float3(floored_idx.x, floored_idx.y, floored_idx.z);
	int base_idx = vol_dim.y * vol_dim.z * floored_idx.x + vol_dim.z * floored_idx.y + floored_idx.z;
	float3 colors[8];
	for (uint8_t i = 0; i < 2; ++i)
	{
		for (uint8_t j = 0; j < 2; ++j)
		{
			for (uint8_t k = 0; k < 2; ++k)
			{
				int vol_idx = base_idx + vol_dim.y * vol_dim.z * i + vol_dim.z * j + k;
				colors[i * 4 + j * 2 + k] = make_float3(tsdf_color[vol_idx].x, tsdf_color[vol_idx].y, tsdf_color[vol_idx].z);
			}
		}
	}
	float3 low = mix(mix(colors[0], colors[4], frac_idx.x), mix(colors[2], colors[6], frac_idx.x), frac_idx.y);
	float3 high = mix(mix(colors[1], colors[5], frac_idx.x), mix(colors[3], colors[7], frac_idx.x), frac_idx.y);
	float3 res = mix(low, high, frac_idx.z);
	return make_uchar3(res.x, res.y, res.z);
}

__device__ void interp_tsdf_cnt(const float3& pos, const float3& vol_start, const float3& voxel, const int3& vol_dim, uint32_t *tsdf_cnt, float *out) {
	float3 idx = (pos - vol_start) / voxel;
	int3 floored_idx = make_int3(floorf(idx.x), floorf(idx.y), floorf(idx.z));
	float3 frac_idx = idx - make_float3(floored_idx.x, floored_idx.y, floored_idx.z);
	int base_idx = vol_dim.y * vol_dim.z * floored_idx.x + vol_dim.z * floored_idx.y + floored_idx.z;

	for (uint8_t m = 0; m < MAX_OBJECTS / 4; ++m)
	{
		float4 diffs[8];
		for (uint8_t i = 0; i < 2; ++i)
		{
			for (uint8_t j = 0; j < 2; ++j)
			{
				for (uint8_t k = 0; k < 2; ++k)
				{
					int vol_idx = base_idx + vol_dim.y * vol_dim.z * i + vol_dim.z * j + k;
					uint4 tmp = *(uint4*)&tsdf_cnt[vol_idx * MAX_OBJECTS + m * 4];
					diffs[i * 4 + j * 2 + k] = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
				}
			}
		}
		float4 low = mix(mix(diffs[0], diffs[4], frac_idx.x), mix(diffs[2], diffs[6], frac_idx.x), frac_idx.y);
		float4 high = mix(mix(diffs[1], diffs[5], frac_idx.x), mix(diffs[3], diffs[7], frac_idx.x), frac_idx.y);
		*(float4*)&out[m * 4] = mix(low, high, frac_idx.z);
	}
	return;
}


