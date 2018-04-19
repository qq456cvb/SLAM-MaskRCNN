#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <thrust/device_vector.h>
#include "helper_math.h"
using namespace std;


__global__ void tsdf_kernel(float *tsdf_diff, int *tsdf_color, int *tsdf_wt, int *tsdf_cls, int *tsdf_cls_cnt,
int vol_dim, float3 *vol_start, float voxel, float miu, float *intrinsic,
uint16_t *depth, uint8_t *color, int *cls, float *extrinsic2init, int width, int height)
{
    uint16_t vol_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t vol_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint16_t vol_idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    float4 pos_homo = make_float4(vol_start[0] + make_float3(vol_idx_x, vol_idx_y, vol_idx_z) * voxel, 1.f);
    float4 proj = make_float4(dot(make_float4(extrinsic2init[0], extrinsic2init[1], extrinsic2init[2], extrinsic2init[3]), pos_homo),
        dot(make_float4(extrinsic2init[4], extrinsic2init[5], extrinsic2init[6], extrinsic2init[7]), pos_homo),
        dot(make_float4(extrinsic2init[8], extrinsic2init[9], extrinsic2init[10], extrinsic2init[11]), pos_homo),
        1.f);
    float screen_x = dot(make_float4(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]), proj);
    float screen_y = dot(make_float4(intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7]), proj);
    float screen_z = dot(make_float4(intrinsic[8], intrinsic[9], intrinsic[10], intrinsic[11]), proj);

    screen_x /= screen_z;
    screen_y /= screen_z;
    int x = __float2int_rd(screen_x);
    int y = __float2int_rd(screen_y);

    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int img_idx = y * width + x;
    if (depth[img_idx] == 0) return;
    float diff = depth[img_idx] / 5000.f - ((float*)&proj)[2];
    if (diff <= -miu) return;
    if (diff > miu) diff = miu;
    diff /= miu;

    uint16_t weight = 1;
    int vol_idx = vol_dim * vol_dim * vol_idx_x + vol_dim * vol_idx_y + vol_idx_z;
    tsdf_diff[vol_idx] = (tsdf_diff[vol_idx] * tsdf_wt[vol_idx] + weight * diff) / (tsdf_wt[vol_idx] + weight);

    for (int i = 0; i < 3; i++) {
        tsdf_color[vol_idx * 3 + i] = (tsdf_color[vol_idx * 3 + i] * tsdf_wt[vol_idx] + weight * color[img_idx * 3 + i]) / (tsdf_wt[vol_idx] + weight);
    }
    tsdf_wt[vol_idx] += weight;
    if (tsdf_cls_cnt[vol_idx] == 0) {
        tsdf_cls[vol_idx] = cls[img_idx];
        tsdf_cls_cnt[vol_idx] = 1;
    } else {
        if (tsdf_cls[vol_idx] == cls[img_idx]) {
            tsdf_cls_cnt[vol_idx]++;
        } else {
            tsdf_cls_cnt[vol_idx]--;
        }
    }
}


void tsdf(float *tsdf_diff, int *tsdf_color, int *tsdf_wt, int *tsdf_cls, int *tsdf_cls_cnt,
int vol_dim, float *vol_start, float voxel, float miu, float *intrinsic,
uint16_t *depth, uint8_t *color, int *cls, float *extrinsic2init, int width, int height)
{
//  cout << vol_dim << endl;
//  for (int i = 0; i < 3; i++) {
//    cout << vol_start[i] << ", ";
//  }
//  cout << endl;
//  cout << voxel << endl;
//  cout << miu << endl;
//  for (int i = 0; i < 4; i++) {
//    for (int j = 0; j < 4; j++)
//      cout << intrinsic[i * 4 + j] << ",";
//    cout << endl;
//  }

    thrust::device_vector<float> tsdf_diff_d(tsdf_diff, tsdf_diff + vol_dim * vol_dim * vol_dim);
    thrust::device_vector<int> tsdf_color_d(tsdf_color, tsdf_color + 3 * vol_dim * vol_dim * vol_dim);
    thrust::device_vector<int> tsdf_wt_d(tsdf_wt, tsdf_wt + vol_dim * vol_dim * vol_dim);
    thrust::device_vector<int> tsdf_cls_d(tsdf_cls, tsdf_cls + vol_dim * vol_dim * vol_dim);
    thrust::device_vector<int> tsdf_cls_cnt_d(tsdf_cls_cnt, tsdf_cls_cnt + vol_dim * vol_dim * vol_dim);
    thrust::device_vector<float> vol_start_d(vol_start, vol_start + 3);
    thrust::device_vector<float> intrinsic_d(intrinsic, intrinsic + 16);
    thrust::device_vector<uint16_t> depth_d(depth, depth + width * height);
    thrust::device_vector<uint8_t> color_d(color, color + 3 * width * height);
    thrust::device_vector<int> cls_d(cls, cls + width * height);
    thrust::device_vector<float> extrinsic2init_d(extrinsic2init, extrinsic2init + 16);

    tsdf_kernel<<<dim3(vol_dim / 8, vol_dim / 8, vol_dim / 8), dim3(8, 8, 8)>>> (
    thrust::raw_pointer_cast(tsdf_diff_d.data()),
    thrust::raw_pointer_cast(tsdf_color_d.data()),
    thrust::raw_pointer_cast(tsdf_wt_d.data()),
    thrust::raw_pointer_cast(tsdf_cls_d.data()),
    thrust::raw_pointer_cast(tsdf_cls_cnt_d.data()),
    vol_dim,
    (float3*)thrust::raw_pointer_cast(vol_start_d.data()),
    voxel,
    miu,
    thrust::raw_pointer_cast(intrinsic_d.data()),
    thrust::raw_pointer_cast(depth_d.data()),
    thrust::raw_pointer_cast(color_d.data()),
    thrust::raw_pointer_cast(cls_d.data()),
    thrust::raw_pointer_cast(extrinsic2init_d.data()),
    width,
    height);

    cudaMemcpy(tsdf_diff, thrust::raw_pointer_cast(tsdf_diff_d.data()), vol_dim * vol_dim * vol_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsdf_wt, thrust::raw_pointer_cast(tsdf_wt_d.data()), vol_dim * vol_dim * vol_dim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsdf_color, thrust::raw_pointer_cast(tsdf_color_d.data()), 3 * vol_dim * vol_dim * vol_dim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsdf_cls, thrust::raw_pointer_cast(tsdf_cls_d.data()), vol_dim * vol_dim * vol_dim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tsdf_cls_cnt, thrust::raw_pointer_cast(tsdf_cls_cnt_d.data()), vol_dim * vol_dim * vol_dim * sizeof(int), cudaMemcpyDeviceToHost);
  // dim3 dimBlock(256, 1, 1);
  // dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
  
  // kernel<<<dimGrid, dimBlock>>>
  //   (vec, scalar, num_elements);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }
}