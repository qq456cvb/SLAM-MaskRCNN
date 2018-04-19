#include "tsdf.cuh"
#include "utils.cuh"
#include <thrust/device_vector.h>
#include "helper_math.h"
#include "configuration.h"
#include <unordered_map>
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

	// TODO: bilinear interpolating
	int x = __float2int_rd(screen_x);
	int y = __float2int_rd(screen_y);

	if (x < 0 || x >= width || y < 0 || y >= height) return;
	int img_idx = y * width + x;
	if (depth[img_idx] == 0) return;
	float diff = depth[img_idx] / 5000.f - proj.z;
	if (diff <= -miu) return;
	if (diff > miu) diff = miu;
	diff = diff / miu;

	uint16_t weight = 1;
	int vol_idx = vol_dim[1] * vol_dim[2] * vol_idx_x + vol_dim[2] * vol_idx_y + vol_idx_z;
	tsdf_diff[vol_idx] = (tsdf_diff[vol_idx] * tsdf_wt[vol_idx] + weight * diff) / (tsdf_wt[vol_idx] + weight);
	/*if (diff < 0.99f) {
		for (int i = 0; i < 3; i++) {
			tsdf_color[vol_idx * 3 + i] = (tsdf_color[vol_idx * 3 + i] * tsdf_wt[vol_idx] + weight * color[img_idx * 3 + i]) / (tsdf_wt[vol_idx] + weight);
		}
		tsdf_cnt[vol_idx * MAX_OBJECTS + mask[img_idx]]++;
	}*/
	
	for (int i = 0; i < 3; i++) {
		tsdf_color[vol_idx * 3 + i] = (tsdf_color[vol_idx * 3 + i] * tsdf_wt[vol_idx] + weight * color[img_idx * 3 + i]) / (tsdf_wt[vol_idx] + weight);
	}
	tsdf_cnt[vol_idx * MAX_OBJECTS + mask[img_idx]]++;
	tsdf_wt[vol_idx] += weight;
	
}

__global__ void back_proj_kernel(float *s2w, float3 *vol_start, float3 *vol_end, float3 *voxel,
	int3 *vol_dim, float *tsdf_diff, uint32_t *tsdf_cnt,
	int width, int height, float *probs, bool *box_mask)
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

	float t = tnear + 1e-6f;
	tfar -= 1e-6f;
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
			for (int i = 0; i < MAX_OBJECTS; i++)
			{
				if (probs[(idx_y * width + idx_x) * MAX_OBJECTS + i] > 1e-6f)
				{
					box_mask[(idx_y * width + idx_x) * MAX_OBJECTS + i] = true;
				}
			}
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
	/*std::cout << intrinsic_ << std::endl;
	std::cout << intrinsic_inv_ << std::endl;*/
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
	free_cuda_vars();
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
		/*std::cout << tl << std::endl;
		std::cout << br << std::endl;*/
		mean_depth_ = mean_depth;

		float half_side = sqrt(pow(tl.at<float>(0, 0) - br.at<float>(0, 0), 2) + pow(tl.at<float>(1, 0) - br.at<float>(1, 0), 2)) / 2;
		cv::Mat center = (tl + br) / 2;
		vol_start_ = cv::Vec3f((float*)center.data) - cv::Vec3f(half_side, half_side, half_side);
		vol_end_ = cv::Vec3f((float*)center.data) + cv::Vec3f(half_side, half_side, half_side);
		cv::divide(vol_end_ - vol_start_, vol_dim_ - cv::Vec3i(1, 1, 1), vol_res_);

		miu_[0] = 5 * vol_res_[0];
		auto size = vol_dim_[0] * vol_dim_[1] * vol_dim_[2];
		tsdf_diff_ = new float[size] {};
		for (int i = 0; i < size; ++i)
		{
			tsdf_diff_[i] = float(miu_[0]);
		}
		tsdf_wt_ = new int[size] {};
		tsdf_color_ = new uint8_t[size * 3] {};
		tsdf_cnt_ = new uint32_t[size * MAX_OBJECTS] {};
		probs = new float[depth.cols * depth.rows * MAX_OBJECTS]{};
		box_mask = new bool[depth.cols * depth.rows * MAX_OBJECTS]{};
		
		init_cuda_vars(depth.cols, depth.rows);
		parse_frame(depth, color, masks, extrinsic, mean_depth);
	}
	else {
		
		cv::Mat extrinsic2init = extrinsic * init_extrinsic_inv_;
		launch_kernel(depth, color, masks, extrinsic2init);

		n_obs_++;
		//cudaMemcpy(tsdf_color_, tsdf_color_d, 3 * vol_dim_[0] * vol_dim_[1] * vol_dim_[2] * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		//cv::Mat tsdf(4096, 4096, CV_8UC3, tsdf_color_);
		////cv::Mat tsdf(4096, 4096, CV_32F, tsdf_diff_);
		//cv::resize(tsdf, tsdf, cv::Size(512, 512));
		//cv::imshow("test", tsdf);
		//cv::waitKey(0);
	}
}

void TSDF::init_cuda_vars(int width, int height) {
	auto size = vol_dim_[0] * vol_dim_[1] * vol_dim_[2];

	cudaMalloc(&probs_d, width * height * MAX_OBJECTS * sizeof(float));
	cudaMemset(probs_d, 0, width * height * MAX_OBJECTS * sizeof(float));

	cudaMalloc(&box_mask_d, width * height * MAX_OBJECTS * sizeof(bool));
	cudaMemset(box_mask_d, 0, width * height * MAX_OBJECTS * sizeof(bool));

	cudaMalloc(&s2w_d, 16 * sizeof(float));
	cudaMemcpy(s2w_d, intrinsic_inv_.data, 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&tsdf_diff_d, size * sizeof(float));
	thrust::device_ptr<float> dev_ptr(tsdf_diff_d);
	thrust::fill(dev_ptr, dev_ptr + size, miu_[0]);

	cudaMalloc(&tsdf_color_d, size * 3 * sizeof(uint8_t));
	cudaMemset(tsdf_color_d, 0, size * 3 * sizeof(uint8_t));

	cudaMalloc(&tsdf_cnt_d, size * MAX_OBJECTS * sizeof(uint32_t));
	cudaMemset(tsdf_cnt_d, 0, size * MAX_OBJECTS * sizeof(uint32_t));

	cudaMalloc(&tsdf_wt_d, size * sizeof(int));
	cudaMemset(tsdf_wt_d, 0, size * sizeof(int));

	cudaMalloc(&vol_start_d, 3 * sizeof(float));
	cudaMemcpy(vol_start_d, vol_start_.val, 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&vol_end_d, 3 * sizeof(float));
	cudaMemcpy(vol_end_d, vol_end_.val, 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&vol_dim_d, 3 * sizeof(int));
	cudaMemcpy(vol_dim_d, vol_dim_.val, 3 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&voxel_d, 3 * sizeof(float));
	cudaMemcpy(voxel_d, vol_res_.val, 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&intrinsic_d, 16 * sizeof(float));
	cudaMemcpy(intrinsic_d, intrinsic_.data, 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&depth_d, width * height * sizeof(uint16_t));
	cudaMalloc(&color_d, width * height * 3 * sizeof(uint8_t));
	cudaMalloc(&mask_d, width * height * sizeof(uint8_t));
	cudaMalloc(&extrinsic2init_d, 16 * sizeof(float));
}

void TSDF::free_cuda_vars() {
	cudaFree(probs_d);
	cudaFree(box_mask_d);
	cudaFree(s2w_d);
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
}

void TSDF::filter_overlaps(float *probs, int width, int height, cv::Mat& mask, bool *box_mask) {
	double max_double;
	cv::minMaxLoc(mask, nullptr, &max_double);
	int max_obj_now = static_cast<int>(max_double) + 1;

	float assignments[MAX_OBJECTS][MAX_OBJECTS] = {0};
	uint32_t cnts[MAX_OBJECTS][MAX_OBJECTS] = { 0 };
	uint8_t *mask_ptr = mask.data;
	for (int i = 0; i < width * height; i++)
	{
		if (mask_ptr[i] > 0)
		{
			for (int j = 1; j < MAX_OBJECTS; j++)
			{
				assignments[mask_ptr[i]][j] += log(max(probs[i * MAX_OBJECTS + j] / n_obs_, Configuration::prior_mrcnn_err_rate));
				cnts[mask_ptr[i]][j]++;
			}
		}
		for (size_t n = 1; n < MAX_OBJECTS; n++)
		{
			// filter valid probs
			if (box_mask[i * MAX_OBJECTS + n]) {
				for (size_t m = 1; m < max_obj_now; m++)
				{
					if (mask_ptr[i] == m) continue;
					assignments[m][n] += log(max(1.f - probs[i * MAX_OBJECTS + n] / n_obs_, Configuration::prior_mrcnn_err_rate));
					cnts[m][n]++;
				}
			}
		}
	}
	std::unordered_map<uint8_t, uint8_t> assign_map, assign_map_rev;
	std::unordered_map<uint8_t, float> assign_map_prob;
	for (int i = 1; i < max_obj_now; i++)
	{
		int max_j = -1;
		float max_prob = 0;
		for (int j = 1; j < MAX_OBJECTS; j++)
		{
			float prob = ((cnts[i][j] == 0) ? 0 : exp(assignments[i][j] / cnts[i][j]));
			if (prob > max_prob) {
				max_j = j;
				max_prob = prob;
			}
		}
		if (max_prob > 3 * Configuration::prior_mrcnn_err_rate)
		{
			printf("current object %d assigned to previous object %d with prob %f\n", i, max_j, max_prob);
			if (assign_map.find(max_j) == assign_map.end())
			{
				assign_map[max_j] = i;
				assign_map_prob[max_j] = max_prob;
			}
			else {
				if (assign_map_prob[max_j] < max_prob)
				{
					assign_map[max_j] = i;
					assign_map_prob[max_j] = max_prob;
				}
			}
		}
	}
	for (auto it : assign_map) {
		assign_map_rev[it.second] = it.first;
		std::cout << (int)it.first << ", " << (int)it.second << std::endl;
	}
	
	std::unordered_map<uint8_t, uint8_t> extra_assign;
	for (int i = 0; i < width * height; i++)
	{
		if (assign_map_rev.find(mask_ptr[i]) != assign_map_rev.end())
		{
			mask_ptr[i] = assign_map_rev[mask_ptr[i]];
		}
		else if (mask_ptr[i] > 0) {
			if (extra_assign.find(mask_ptr[i]) == extra_assign.end())
			{
				extra_assign[mask_ptr[i]] = num_objs;
				if (num_objs == 16)
				{
					std::cout << num_objs << ", " <<  (int)mask_ptr[i] << std::endl;
				}
				mask_ptr[i] = num_objs;
				num_objs++;
			}
			else {
				mask_ptr[i] = extra_assign[mask_ptr[i]];
			}
		}
	}
	/*int total = width * height;
	cv::Mat test(height, width, CV_32FC1, cv::Scalar(0));
	float *ptr = (float*)test.data;
	for (int i = 0; i < total; i++)
	{
		ptr[i] = (mask_ptr[i] == 16) ? 1.f : 0.f;
	}
	cv::imshow("after", test);
	cv::waitKey(0);*/
	/*for (int i = 0; i < total; i++)
	{
		ptr[i] = (mask_ptr[i] == 2) ? 1.f : 0.f;
	}
	cv::imshow("after", test);
	cv::waitKey();*/
	printf("num objs %d\n", num_objs);
	//cv::minMaxLoc(mask, nullptr, &max_double);
	//max_obj_now = static_cast<int>(max_double) + 1;
	//num_objs = max(num_objs, max_obj_now);
}

void TSDF::launch_kernel(const cv::Mat& depth, const cv::Mat& color, cv::Mat& masks, const cv::Mat& extrinsic2init) {
	int width = depth.cols;
	int height = depth.rows;

	cudaMemcpy(depth_d, depth.data, width * height * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(color_d, color.data, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(extrinsic2init_d, extrinsic2init.data, 16 * sizeof(float), cudaMemcpyHostToDevice);


	if (n_obs_ > 0)
	{
		cudaMemset(probs_d, 0, width * height * MAX_OBJECTS * sizeof(float));
		cudaMemset(box_mask_d, 0, width * height * MAX_OBJECTS * sizeof(bool));
		cv::Mat s2w = extrinsic2init.inv() * intrinsic_inv_;
		cudaMemcpy(s2w_d, s2w.data, 16 * sizeof(float), cudaMemcpyHostToDevice);

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
			probs_d,
			box_mask_d
			);
		
		cudaMemcpy(probs, probs_d, MAX_OBJECTS * width * height * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(box_mask, box_mask_d, MAX_OBJECTS * width * height * sizeof(bool), cudaMemcpyDeviceToHost);
		
		filter_overlaps(probs, width, height, masks, box_mask);
		
	}
	else {
		double max_double;
		cv::minMaxLoc(masks, nullptr, &max_double);
		num_objs = static_cast<int>(max_double) + 1;
		printf("max_objs:%d\n", num_objs);
	}

	cudaMemcpy(mask_d, masks.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

	tsdf_kernel << <dim3((vol_dim_[0] - 1) / 8 + 1, (vol_dim_[1] - 1) / 8 + 1, (vol_dim_[2] - 1) / 8 + 1), dim3(8, 8, 8) >> > (
		tsdf_diff_d,
		tsdf_color_d,
		tsdf_cnt_d,
		tsdf_wt_d,
		vol_dim_d,
		vol_start_d,
		voxel_d,
		(float)miu_[0],
		intrinsic_d,
		depth_d,
		color_d,
		mask_d,
		extrinsic2init_d,
		width,
		height
	);

	/*cudaMemcpy(tsdf_diff_, tsdf_diff_d, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_wt_, tsdf_wt_d, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_color_, tsdf_color_d, 3 * size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_cnt_, tsdf_cnt_d, MAX_OBJECTS * size * sizeof(uint32_t), cudaMemcpyDeviceToHost);*/

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
