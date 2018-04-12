#include "tsdf.cuh"
#include "utils.cuh"
#include <thrust/device_vector.h>
#include "helper_math.h"
#include <cuda_runtime.h>
#include <vector_functions.h>

__global__ void tsdf_kernel(float *tsdf_diff, uint8_t *tsdf_color, int *tsdf_wt,
	int *vol_dim, float *vol_start, float *voxel, float miu, float *intrinsic,
	uint16_t *depth, uint8_t *color, float *extrinsic2init, int width, int height)
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
void TSDF::parse_frame(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsic, float mean_depth) {
	// init bounding box
	if (!init_) {
		// TODO: decide the best scale factor to transform volume into unit box
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
		
		parse_frame(depth, color, extrinsic, mean_depth);
	}
	else {
		cv::Mat extrinsic2init = extrinsic * init_extrinsic_inv_;
		launch_kernel(depth, color, extrinsic2init);
		//cv::Mat tsdf(4096, 4096, CV_32SC3, tsdf_color_);
		//tsdf.convertTo(tsdf, CV_8UC3);
		////cv::Mat tsdf(4096, 4096, CV_32F, tsdf_diff_);
		//cv::resize(tsdf, tsdf, cv::Size(512, 512));
		//cv::imshow("test", tsdf);
		//cv::waitKey(0);
	}
}

void TSDF::launch_kernel(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsic2init) {
	auto size = vol_dim_[0] * vol_dim_[1] * vol_dim_[2];
	int width = depth.cols;
	int height = depth.rows;
	thrust::device_vector<float> tsdf_diff_d(tsdf_diff_, tsdf_diff_ + size);
	thrust::device_vector<uint8_t> tsdf_color_d(tsdf_color_, tsdf_color_ + 3 * size);
	thrust::device_vector<int> tsdf_wt_d(tsdf_wt_, tsdf_wt_ + size);
	thrust::device_vector<float> vol_start_d((float*)vol_start_.val, (float*)vol_start_.val + 3);
	thrust::device_vector<int> vol_dim_d((int*)vol_dim_.val, (int*)vol_dim_.val + 3);
	thrust::device_vector<float> voxel_d((float*)vol_res_.val, (float*)vol_res_.val + 3);
	thrust::device_vector<float> intrinsic_d((float*)intrinsic_.data, (float*)intrinsic_.data + 16);
	thrust::device_vector<uint16_t> depth_d((uint16_t*)depth.data, (uint16_t*)depth.data + width * height);
	thrust::device_vector<uint8_t> color_d((uint8_t*)color.data, (uint8_t*)color.data + 3 * width * height);
	thrust::device_vector<float> extrinsic2init_d((float*)extrinsic2init.data, (float*)extrinsic2init.data + 16);

	tsdf_kernel << <dim3((vol_dim_[0] - 1) / 8 + 1, (vol_dim_[1] - 1) / 8 + 1, (vol_dim_[2] - 1) / 8 + 1), dim3(8, 8, 8) >> > (
		thrust::raw_pointer_cast(tsdf_diff_d.data()),
		thrust::raw_pointer_cast(tsdf_color_d.data()),
		thrust::raw_pointer_cast(tsdf_wt_d.data()),
		thrust::raw_pointer_cast(vol_dim_d.data()),
		thrust::raw_pointer_cast(vol_start_d.data()),
		thrust::raw_pointer_cast(voxel_d.data()),
		(float)miu_.val[0],
		thrust::raw_pointer_cast(intrinsic_d.data()),
		thrust::raw_pointer_cast(depth_d.data()),
		thrust::raw_pointer_cast(color_d.data()),
		thrust::raw_pointer_cast(extrinsic2init_d.data()),
		width,
		height
	);

	cudaMemcpy(tsdf_diff_, thrust::raw_pointer_cast(tsdf_diff_d.data()), size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_wt_, thrust::raw_pointer_cast(tsdf_wt_d.data()), size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tsdf_color_, thrust::raw_pointer_cast(tsdf_color_d.data()), 3 * size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
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