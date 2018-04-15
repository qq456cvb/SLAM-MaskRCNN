#include "viewer.cuh"
#include <thrust/device_vector.h>
#include "helper_math.h"
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "utils.cuh"

template<typename T>
T* malloc_and_cpy(T *host_ptr, size_t cnt) {
	T *device_ptr;
	cudaMalloc(&device_ptr, cnt * sizeof(T));
	cudaMemcpy(device_ptr, host_ptr, cnt * sizeof(T), cudaMemcpyHostToDevice);
	return device_ptr;
}

__global__ void show_tsdf_kernel(float *s2w, float3 *c, float3 *vol_start, float3 *vol_end, float3 *voxel,
	int3 *vol_dim, float *tsdf_diff, uchar3 *tsdf_color, uint32_t *tsdf_cnt,
	int width, int height, uchar3 *output)
{
	uint16_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx_x >= width) return;
	if (idx_y >= height) return;

	float4 screen_pos = make_float4(idx_x, idx_y, 1.f, 1.f);
	float3 target = make_float3(dot(make_float4(s2w[0], s2w[1], s2w[2], s2w[3]), screen_pos),
		dot(make_float4(s2w[4], s2w[5], s2w[6], s2w[7]), screen_pos),
		dot(make_float4(s2w[8], s2w[9], s2w[10], s2w[11]), screen_pos)
	);

	float3 d = normalize(target - c[0]);
	float3 inv_d = 1.f / d;
	float3 tbot = inv_d * (vol_start[0] - c[0]);
	float3 ttop = inv_d * (vol_end[0] - c[0]);

	float3 tmin = make_float3(min(ttop.x, tbot.x), min(ttop.y, tbot.y), min(ttop.z, tbot.z));
	float tnear = max(max(tmin.x, tmin.y), tmin.z);
	tnear = max(tnear, 0.01f);

	float3 tmax = make_float3(max(ttop.x, tbot.x), max(ttop.y, tbot.y), max(ttop.z, tbot.z));
	float tfar = min(min(tmax.x, tmax.y), tmax.z);
	tfar = min(tfar, 100.f);
	if (tnear > tfar) return;

	float t = tnear + 1e-6f;
	tfar -= 1e-6f;
	float f_tt = 0;
	float stepsize = voxel[0].x;
	float f_t = interp_tsdf_diff(c[0] + t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_diff);
	if (f_t > 0) {
		for (; t < tfar; t += stepsize)
		{
			f_tt = interp_tsdf_diff(c[0] + t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_diff);
			if (f_tt < 0.f)
			{
				break;
			}
			if (f_tt < voxel[0].x / 2.f)
			{
				stepsize = voxel[0].x / 4.f;
			}
			f_t = f_tt;
		}
		if (f_tt < 0.f)
		{
			t += stepsize * f_tt / (f_t - f_tt);
			output[(idx_y * width + idx_x)] = interp_tsdf_color(c[0] + t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_color);
			/*float cnts[MAX_OBJECTS];
			interp_tsdf_cnt(c[0] + t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_cnt, cnts);
			float max_cnt = 0;
			uint8_t obj_idx = 0;
			for (uint8_t k = 0; k < MAX_OBJECTS; k++)
			{
				if (cnts[k] > max_cnt) {
					max_cnt = cnts[k];
					obj_idx = k;
				}
			}
			output[(idx_y * width + idx_x)] = make_uchar3(obj_idx * 20, obj_idx * 20, obj_idx * 20);*/
		}
	}
}

Viewer::Viewer(int width, int height) : width_(width), height_(height) {
	cudaMalloc(&s2w_d, 16 * sizeof(float));
	cudaMalloc(&c_d, 3 * sizeof(float));
	cudaMalloc(&output_d, width * height * sizeof(uchar3));
}

Viewer::~Viewer() {
	cudaFree(s2w_d);
	cudaFree(c_d);
	cudaFree(output_d);
}

cv::Mat Viewer::show_tsdf(const TSDF& tsdf, float angle, float dist) {
	cv::Mat img(height_, width_, CV_8UC3, cv::Scalar(0));

	float rot[16] = { std::cosf(angle), 0, -std::sinf(angle), dist * std::sinf(angle), 0, 1, 0, 0, std::sinf(angle), 0, std::cosf(angle), dist - dist * std::cosf(angle), 0, 0, 0, 1 };
	cv::Mat extrinsic(4, 4, CV_32F, rot);
	cv::Mat s2w = extrinsic * tsdf.get_intrinsic_inv();

	float center[3] = { 0 };
	center[0] = (dist + 0.5f) * std::sinf(angle);
	center[2] = (dist + 0.5f) - (dist + 0.5f) * std::cosf(angle);

	cudaMemcpy(s2w_d, s2w.data, 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, center, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(output_d, 0, width_ * height_ * sizeof(uchar3));

	show_tsdf_kernel << <dim3((width_ - 1) / 32 + 1, (height_ - 1) / 32 + 1, 1), dim3(32, 32, 1) >> > (
		s2w_d,
		(float3*)c_d,
		(float3*)tsdf.vol_start_d,
		(float3*)tsdf.vol_end_d,
		(float3*)tsdf.voxel_d,
		(int3*)tsdf.vol_dim_d,
		tsdf.tsdf_diff_d,
		(uchar3*)tsdf.tsdf_color_d,
		tsdf.tsdf_cnt_d,
		width_,
		height_,
		output_d
		);
	cudaMemcpy(img.data, output_d, width_ * height_ * sizeof(uchar3), cudaMemcpyDeviceToHost);

	auto error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::stringstream strstr;
		strstr << "run_kernel launch failed" << std::endl;
		strstr << cudaGetErrorString(error);
		throw strstr.str();
	}
	cv::imshow("img", img);
	cv::waitKey(10);
	return img;
}