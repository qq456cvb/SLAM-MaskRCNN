#include "tsdf.cuh"
#include "utils.cuh"
#include <thrust/device_vector.h>
#include "helper_math.h"
#include "configuration.h"
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

template<int num_objs>
__global__ void proj_kernel(float prior_mrcnn_err_rate, int n_obs, float *tsdf_diff, uint32_t *tsdf_cnt,
	int *vol_dim, float *vol_start, float *voxel, float miu, float *intrinsic, float *extrinsic2init,
	int width, int height, uint16_t *depth, float *probs, bool *box_mask)
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
	int vol_idx = vol_dim[1] * vol_dim[2] * vol_idx_x + vol_dim[2] * vol_idx_y + vol_idx_z;

	/*for (int i = 0; i < num_objs; i++) {
		probs[img_idx * MAX_OBJECTS + i] += max(prior_mrcnn_err_rate, (float)tsdf_cnt[vol_idx * MAX_OBJECTS + i] / n_obs);
	}*/
		
	
	if (diff < 0.99f) {
		box_mask[img_idx * MAX_OBJECTS] = true;
		for (int i = 0; i < num_objs; i++)
		{
			probs[img_idx * MAX_OBJECTS + i] = (float)tsdf_cnt[vol_idx * MAX_OBJECTS + i] / n_obs;
			//atomicAdd(&probs[img_idx * MAX_OBJECTS + i], (float)tsdf_cnt[vol_idx * MAX_OBJECTS + i] / n_obs);
			if (tsdf_cnt[vol_idx * MAX_OBJECTS + i] > 0)
			{
				box_mask[img_idx * MAX_OBJECTS + i] = true;
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
	float assignments[MAX_OBJECTS][MAX_OBJECTS] = {0};
	uint32_t cnts[MAX_OBJECTS][MAX_OBJECTS] = { 0 };
	uint8_t *mask_ptr = mask.data;
	for (int i = 0; i < width * height; i++)
	{
		if (mask_ptr[i] > 0)
		{
			for (int j = 1; j < MAX_OBJECTS; j++)
			{
				// filter valid probs
				if (box_mask[i * MAX_OBJECTS]) {
					assignments[mask_ptr[i]][j] += log(max(probs[i * MAX_OBJECTS + j], Configuration::prior_mrcnn_err_rate));
					cnts[mask_ptr[i]][j]++;
				}
			}
		}
		for (size_t n = 1; n < MAX_OBJECTS; n++)
		{
			// filter valid probs
			if (box_mask[i * MAX_OBJECTS] && box_mask[i * MAX_OBJECTS + n]) {
				for (size_t m = 1; m < MAX_OBJECTS; m++)
				{
					if (mask_ptr[i] == m) continue;
					assignments[m][n] += log(max(1.f - probs[i * MAX_OBJECTS + n], Configuration::prior_mrcnn_err_rate));
					cnts[m][n]++;
				}
			}
		}
	}
	for (int i = 1; i < MAX_OBJECTS; i++)
	{
		for (int j = 1; j < MAX_OBJECTS; j++)
		{
			float prob = ((cnts[i][j] == 0) ? 0 : exp(assignments[i][j] / cnts[i][j]));
			if (prob > 0.2f)
			{
				printf("current object %d assigned to previous object %d with prob %f\n", i, j, prob);
			}
		}
	}
}

void TSDF::launch_kernel(const cv::Mat& depth, const cv::Mat& color, cv::Mat& masks, const cv::Mat& extrinsic2init) {
	int width = depth.cols;
	int height = depth.rows;

	cudaMemcpy(depth_d, depth.data, width * height * sizeof(uint16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(color_d, color.data, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, masks.data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(extrinsic2init_d, extrinsic2init.data, 16 * sizeof(float), cudaMemcpyHostToDevice);


	if (n_obs_ > 0)
	{
		cudaMemset(probs_d, 0, width * height * MAX_OBJECTS * sizeof(float));
		cudaMemset(box_mask_d, 0, width * height * MAX_OBJECTS * sizeof(bool));
		proj_kernel<MAX_OBJECTS><< <dim3((vol_dim_[0] - 1) / 8 + 1, (vol_dim_[1] - 1) / 8 + 1, (vol_dim_[2] - 1) / 8 + 1), dim3(8, 8, 8) >> > (
			Configuration::prior_mrcnn_err_rate,
			n_obs_,
			tsdf_diff_d,
			tsdf_cnt_d,
			vol_dim_d,
			vol_start_d,
			voxel_d,
			(float)miu_[0],
			intrinsic_d,
			extrinsic2init_d,
			width,
			height,
			depth_d,
			probs_d,
			box_mask_d
			);

		cudaMemcpy(probs, probs_d, MAX_OBJECTS * width * height * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(box_mask, box_mask_d, MAX_OBJECTS * width * height * sizeof(bool), cudaMemcpyDeviceToHost);
		int total = width * height;
		cv::Mat test(height, width, CV_32FC1, cv::Scalar(0));
		float *ptr = (float*)test.data;
		for (int i = 0; i < total; i++)
		{
			if (box_mask[i * MAX_OBJECTS + 1])
			{
				assert(box_mask[i * MAX_OBJECTS]);
			}
			ptr[i] = (masks.data[i] == 2 && box_mask[i * MAX_OBJECTS]) ? 1.f : 0.f;
		}
		cv::imshow("test", test);
		cv::waitKey();
		filter_overlaps(probs, width, height, masks, box_mask);
		int a = 0;
	}
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