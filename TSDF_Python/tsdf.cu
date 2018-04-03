#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// __global__ void kernel
// (double *vec, double scalar, int num_elements)
// {
//   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < num_elements) {
//     vec[idx] = vec[idx] * scalar;
//   }
// }


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