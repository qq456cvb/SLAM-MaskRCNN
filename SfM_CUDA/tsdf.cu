#include "tsdf.cuh"
#include "utils.cuh"
#include <thrust/device_vector.h>
#include "helper_math.h"
#include <cuda_runtime.h>
#include <vector_functions.h>

template<typename T>
T* malloc_and_cpy(T *host_ptr, size_t cnt) {
	T *device_ptr;
	cudaMalloc(&device_ptr, cnt * sizeof(T));
	cudaMemcpy(device_ptr, host_ptr, cnt * sizeof(T), cudaMemcpyHostToDevice);
	return device_ptr;
}

__global__ void tsdf_kernel(float *tsdf_diff, uint8_t *tsdf_color, uint32_t *tsdf_cnt, int *tsdf_wt,
	int *vol_dim, float *vol_start, float *voxel, float miu, float *intrinsic,
	uint16_t *depth, uint8_t *color, uint8_t *mask, float *extrinsic2init, int width, int height)
{
	uint16_t vol_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t vol_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	uint16_t vol_idx_z = blockIdx.z * blockDim.z + threadIdx.z;
	if (vol_idx_x >= vol_dim[0] || vol_idx_y >= vol_dim[1] || vol_idx_z >= vol_dim[2])
	{
		return;
	}

	float4 pos_homo = make_float4(make_float3(vol_start[0], vol_start[1], vol_start[2]) + make_float3(vol_idx_x, vol_idx_y, vol_idx_z) * make_float3(voxel[0], voxel[1], voxel[2]), 1.f);
	float3 proj = make_float3(dot(make_float4(extrinsic2init[0], extrinsic2init[1], extrinsic2init[2], extrinsic2init[3]), pos_homo),
		dot(make_float4(extrinsic2init[4], extrinsic2init[5], extrinsic2init[6], extrinsic2init[7]), pos_homo),
		dot(make_float4(extrinsic2init[8], extrinsic2init[9], extrinsic2init[10], extrinsic2init[11]), pos_homo)
	);
	float screen_x = dot(make_float3(intrinsic[0], intrinsic[1], intrinsic[2]), proj);
	float screen_y = dot(make_float3(intrinsic[4], intrinsic[5], intrinsic[6]), proj);
	float screen_z = dot(make_float3(intrinsic[8], intrinsic[9], intrinsic[10]), proj);

	screen_x /= screen_z;
	screen_y /= screen_z;
	int x = __float2int_rd(screen_x);
	int y = __float2int_rd(screen_y);

	if (x < 0 || x >= width || y < 0 || y >= height) return;
	int img_idx = y * width + x;
	if (depth[img_idx] == 0) return;
	float diff = depth[img_idx] / 5000.f - proj.z;
	if (diff <= -miu) return;
	if (diff > miu) diff = miu;
	diff /= miu;

	uint16_t weight = 1;
	int vol_idx = vol_dim[1] * vol_dim[2] * vol_idx_x + vol_dim[2] * vol_idx_y + vol_idx_z;
	tsdf_diff[vol_idx] = (tsdf_diff[vol_idx] * tsdf_wt[vol_idx] + weight * diff) / (tsdf_wt[vol_idx] + weight);
	for (int i = 0; i < 3; i++) {
		tsdf_color[vol_idx * 3 + i] = (tsdf_color[vol_idx * 3 + i] * tsdf_wt[vol_idx] + weight * color[img_idx * 3 + i]) / (tsdf_wt[vol_idx] + weight);
	}
	tsdf_wt[vol_idx] += weight;
	if (mask[img_idx] > 0) tsdf_cnt[vol_idx * MAX_OBJECTS + mask[img_idx]]++;
}

__global__ void back_proj_kernel(float *s2w, float3 *vol_start, float3 *vol_end, float3 *voxel,
	int3 *vol_dim, float *tsdf_diff, uint32_t *tsdf_cnt,
	int width, int height, float *probs)
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

	float3 d = normalize(target);
	float3 inv_d = 1.f / d;
	float3 tbot = inv_d * (vol_start[0]);
	float3 ttop = inv_d * (vol_end[0]);

	float3 tmin = make_float3(min(ttop.x, tbot.x), min(ttop.y, tbot.y), min(ttop.z, tbot.z));
	float tnear = max(max(tmin.x, tmin.y), tmin.z);
	tnear = max(tnear, 0.01f);

	float3 tmax = make_float3(max(ttop.x, tbot.x), max(ttop.y, tbot.y), max(ttop.z, tbot.z));
	float tfar = min(min(tmax.x, tmax.y), tmax.z);
	tfar = min(tfar, 100.f);
	if (tnear > tfar) return;

	float t = tnear;
	float f_tt = 0;
	float stepsize = voxel[0].x;
	float f_t = interp_tsdf_diff(t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_diff);
	if (f_t > 0) {
		for (; t < tfar; t += stepsize)
		{
			f_tt = interp_tsdf_diff(t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_diff);
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
			interp_tsdf_cnt(t * d, vol_start[0], voxel[0], vol_dim[0], tsdf_cnt, &probs[(idx_y * width + idx_x) * MAX_OBJECTS]);
		}
	}
}

TSDF::TSDF(cv::Scalar intrinsics)
{
	float fx = intrinsics[0];
	float fy = intrinsics[1];
	float cx = intrinsics[2];
	float cy = intrinsics[3];
	this->intrinsic_.at<float>(0, 0) = fx;
	this->intrinsic_.at<float>(1, 1) = fy;
	this->intrinsic_.at<float>(0, 2) = cx;
	this->intrinsic_.at<float>(1, 2) = cy;
	intrinsic_inv_ = intrinsic_.inv();
	std::cout << intrinsic_ << std::endl;
	std::cout << intrinsic_inv_ << std::endl;
}

TSDF::~TSDF()
{
	if (tsdf_diff_)
	{
		delete[] tsdf_diff_;
	}
	
	if (tsdf_color_)
	{
		delete[] tsdf_color_;
	}
	if (tsdf_wt_)
	{
		delete[] tsdf_wt_;
	}
}

// notice: we get tsdf in left-handed coordinates
void TSDF::parse_frame(const cv::Mat& depth, const cv::Mat& color, cv::Mat& masks, const cv::Mat& extrinsic, float mean_depth) {
	// init bounding box
	if (!init_) {
		// TODO: decide the best scale factor to transform volume into unit box
		n_obs_ = 0;
		init_ = true;
		init_extrinsic_inv_ = extrinsic.inv();
		cv::Mat points;
		cv::Mat depth_mask;
		depth.convertTo(depth_mask, CV_8UC1);
		cv::findNonZero(depth_mask, points);
		cv::Rect min_rect = cv::boundingRect(points);

		// we use the diagonal as the volume side
		cv::Mat tl = intrinsic_inv_ * cv::Mat(4, 1, CV_32F, std::vector<float>({ (float)min_rect.tl().x, (float)min_rect.tl().y, 1.0f, 1.0f }).data());
		cv::Mat br = intrinsic_inv_ * cv::Mat(4, 1, CV_32F, std::vector<float>({ (float)min_rect.br().x, (float)min_rect.br().y, 1.0f, 1.0f }).data());
		tl = tl * mean_depth;
		br = br * mean_depth;
		std::cout << tl << std::endl;
		std::cout << br << std::endl;
		mean_depth_ = mean_depth;

		float half_side = sqrt(pow(tl.at<float>(0, 0) - br.at<float>(0, 0), 2) + pow(tl.at<float>(1, 0) - br.at<float>(1, 0), 2)) / 2;
		cv::Mat center = (tl + br) / 2;
		vol_start_ = cv::Vec3f((float*)center.data) - cv::Vec3f(half_side, half_side, half_side);
		vol_end_ = cv::Vec3f((float*)center.data) + cv::Vec3f(half_side, half_side, half_side);
		cv::divide(vol_end_ - vol_start_, vol_dim_ - cv::Vec3i(1, 1, 1), vol_res_);

		miu_ = 5 * vol_res_;
		auto size = vol_dim_[0] * vol_dim_[1] * vol_dim_[2];
		tsdf_diff_ = new float[size] {};
		for (int i = 0; i < size; ++i)
		{
			tsdf_diff_[i] = float(miu_[0]);
		}
		tsdf_wt_ = new int[size] {};
		tsdf_color_ = new uint8_t[size * 3] {};
		tsdf_cnt_ = new uint32_t[size * MAX_OBJECTS] {};
		
		parse_frame(depth, color, masks, extrinsic, mean_depth);
	}
	else {
		
		cv::Mat extrinsic2init = extrinsic * init_extrinsic_inv_;
		launch_kernel(depth, color, masks, extrinsic2init);

		n_obs_++;
		//cv::Mat tsdf(4096, 4096, CV_32SC3, tsdf_color_);
		//tsdf.convertTo(tsdf, CV_8UC3);
		////cv::Mat tsdf(4096, 4096, CV_32F, tsdf_diff_);
		//cv::resize(tsdf, tsdf, cv::Size(512, 512));
		//cv::imshow("test", tsdf);
		//cv::waitKey(0);
	}
}

void TSDF::filter_overlaps(float *probs, int width, int height, cv::Mat& mask) {
	const double mask_failure_prob = 0.05;
	double assignments[MAX_OBJECTS][MAX_OBJECTS] = {0};
	uint32_t cnts[MAX_OBJECTS] = { 0 };
	uint8_t *mask_ptr = mask.data;
	for (int i = 0; i < width * height; i++)
	{
		if (mask_ptr[i] > 0)
		{
			for (int j = 0; j < MAX_OBJECTS; j++)
			{
				/*double prob = (double)probs[i * MAX_OBJECTS + j] / n_obs_;
				if (prob > 0) printf("found %d assigned to previous object %d with prob %f\n", mask_ptr[i], j, probs[i * MAX_OBJECTS + j] / n_obs_);
				assignments[mask_ptr[i]][j] = (log(max((double)probs[i * MAX_OBJECTS + j] / n_obs_, (double)0)) + assignments[mask_ptr[i]][j] * cnts[mask_ptr[i]]) / (cnts[mask_ptr[i]] + 1);*/
				assignments[mask_ptr[i]][j] += log(max((double)probs[i * MAX_OBJECTS + j] / n_obs_, mask_failure_prob));
			}
			cnts[mask_ptr[i]]++;
		}
	}
	for (int i = 0; i < MAX_OBJECTS; i++)
	{
		for (int j = 0; j < MAX_OBJECTS; j++)
		{
			double prob = (cnts[i] == 0 || assignments[i][j] / cnts[i] == 0) ? 0 : exp(assignments[i][j] / cnts[i]);
			printf("current object %d assigned to previous object %d with prob %f\n", i, j, prob);
		}
	}
}

void TSDF::launch_kernel(const cv::Mat& depth, const cv::Mat& color, cv::Mat& masks, const cv::Mat& extrinsic2init) {
	auto size = vol_dim_[0] * vol_dim_[1] * vol_dim_[2];
	int width = depth.cols;
	int height = depth.rows;
	float *tsdf_diff_d = malloc_and_cpy(tsdf_diff_, size);
	uint8_t *tsdf_color_d = malloc_and_cpy(tsdf_color_, 3 * size);
	uint32_t *tsdf_cnt_d = malloc_and_cpy(tsdf_cnt_, size * MAX_OBJECTS);
	int *tsdf_wt_d = malloc_and_cpy(tsdf_wt_, size);
	float *vol_start_d = malloc_and_cpy((float*)vol_start_.val, 3);
	float *vol_end_d = malloc_and_cpy((float*)vol_end_.val, 3);
	int *vol_dim_d = malloc_and_cpy((int*)vol_dim_.val, 3);
	float *voxel_d = malloc_and_cpy((float*)vol_res_.val, 3);
	float *intrinsic_d = malloc_and_cpy((float*)intrinsic_.data, 16);
	uint16_t *depth_d = malloc_and_cpy((uint16_t*)depth.data, width * height);
	uint8_t *color_d = malloc_and_cpy((uint8_t*)color.data, 3 * width * height);
	uint8_t *mask_d = malloc_and_cpy((uint8_t*)masks.data, width * height);
	float *extrinsic2init_d = malloc_and_cpy((float*)extrinsic2init.data, 16);
	

	if (n_obs_ > 0)
	{
		float *probs = new float[width * height * MAX_OBJECTS]{};
		float *probs_d = malloc_and_cpy(probs, width * height * MAX_OBJECTS);
		float *s2w_d = malloc_and_cpy((float*)intrinsic_inv_.data, 16);
		back_proj_kernel << <dim3((width - 1) / 32 + 1, (height - 1) / 32 + 1, 1), dim3(32, 32, 1) >> > (
			s2w_d,
			(float3*)vol_start_d,
			(float3*)vol_end_d,
			(float3*)voxel_d,
			(int3*)vol_dim_d,
			tsdf_diff_d,
			tsdf_cnt_d,
			width,
			height,
			probs_d
			);

		cudaMemcpy(probs, probs_d, MAX_OBJECTS * width * height * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(probs_d);
		cudaFree(s2w_d);

		filter_overlaps(probs, width, height, masks);
	}
	tsdf_kernel << <dim3((vol_dim_[0] - 1) / 8 + 1, (vol_dim_[1] - 1) / 8 + 1, (vol_dim_[2] - 1) / 8 + 1), dim3(8, 8, 8) >> > (
		tsdf_diff_d,
		tsdf_color_d,
		tsdf_cnt_d,
		tsdf_wt_d,
		vol_dim_d,
		vol_start_d,
		voxel_d,
		(float)miu_.val[0],
		intrinsic_d,
		depth_d,
		color_d,
		mask_d,
		extrinsic2init_d,
		width,
		height
	);

	cudaMemcpy(tsdf_diff_, tsdf_diff_d, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_wt_, tsdf_wt_d, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_color_, tsdf_color_d, 3 * size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_cnt_, tsdf_cnt_d, MAX_OBJECTS * size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	

	cudaFree(tsdf_diff_d);
	cudaFree(tsdf_wt_d);
	cudaFree(tsdf_color_d);
	cudaFree(tsdf_cnt_d);
	cudaFree(vol_dim_d);
	cudaFree(vol_start_d);
	cudaFree(vol_end_d);
	cudaFree(voxel_d);
	cudaFree(intrinsic_d);
	cudaFree(depth_d);
	cudaFree(color_d);
	cudaFree(mask_d);
	cudaFree(extrinsic2init_d);
	
	

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::stringstream strstr;
		strstr << "run_kernel launch failed" << std::endl;
		strstr << cudaGetErrorString(error);
		throw strstr.str();
	}
}

uint8_t *TSDF::get_tsdf_color() const {
	return tsdf_color_;
}

float *TSDF::get_tsdf_diff() const {
	return tsdf_diff_;
}

uint32_t *TSDF::get_tsdf_cnt() const {
	return tsdf_cnt_;
}

cv::Vec3i TSDF::get_dim() const {
	return vol_dim_;
}

cv::Vec3f TSDF::get_vol_start() const {
	return vol_start_;
}

cv::Vec3f TSDF::get_vol_end() const {
	return vol_end_;
}

cv::Mat TSDF::get_intrinsic() const {
	return intrinsic_;
}

cv::Mat TSDF::get_intrinsic_inv() const {
	return intrinsic_inv_;
}

cv::Vec3f TSDF::get_voxel() const {
	return vol_res_;
}