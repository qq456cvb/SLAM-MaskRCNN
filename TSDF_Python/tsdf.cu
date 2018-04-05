#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include "helper_math.h"
using namespace std;


__global__ void tsdf_kernel(float *tsdf_diff, int *tsdf_color, int *tsdf_wt,
int vol_dim, float3 *vol_start, float voxel, float miu, float *intrinsic,
uint16_t *depth, uint8_t *color, float *extrinsic2init, int width, int height)
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

    uint16_t weight = 1;
    uint16_t vol_idx = vol_dim * vol_dim * vol_idx_x + vol_dim * vol_idx_y + vol_idx_z;
    tsdf_diff[vol_idx] = (tsdf_diff[vol_idx] * tsdf_wt[vol_idx] + weight * diff) / (tsdf_wt[vol_idx] + weight);
    for (int i = 0; i < 3; i++) {
        tsdf_color[vol_idx * 3 + i] = (tsdf_color[vol_idx * 3 + i] * tsdf_wt[vol_idx] + weight * color[img_idx * 3 + i]) / (tsdf_wt[vol_idx] + weight);
    }
    tsdf_wt[vol_idx] += weight;

}


void tsdf(float *tsdf_diff, int *tsdf_color, int *tsdf_wt, 
int vol_dim, float *vol_start, float voxel, float miu, float *intrinsic, float *intrinsic_inv,
uint16_t *depth, uint8_t *color, float *extrinsic)
{
  cout << vol_dim << endl;
  for (int i = 0; i < 3; i++) {
    cout << vol_start[i] << ", ";
  }
  cout << endl;
  cout << voxel << endl;
  cout << miu << endl;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      cout << intrinsic[i * 4 + j] << ",";
    cout << endl;
  }
  // dim3 dimBlock(256, 1, 1);
  // dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
  
  // kernel<<<dimGrid, dimBlock>>>
  //   (vec, scalar, num_elements);

  // cudaError_t error = cudaGetLastError();
  // if (error != cudaSuccess) {
  //   std::stringstream strstr;
  //   strstr << "run_kernel launch failed" << std::endl;
  //   strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
  //   strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
  //   strstr << cudaGetErrorString(error);
  //   throw strstr.str();
  // }
}