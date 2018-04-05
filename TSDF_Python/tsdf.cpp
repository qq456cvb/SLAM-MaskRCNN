#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

void tsdf(float *tsdf_diff, int *tsdf_color, int *tsdf_wt,
int vol_dim, float *vol_start, float voxel, float miu, float *intrinsic,
uint16_t *depth, uint8_t *color, float *extrinsic2init, int width, int height);

void tsdf_update(pybind11::array_t<float> tsdf_diff, pybind11::array_t<int> tsdf_color, pybind11::array_t<int> tsdf_wt, 
int vol_dim, pybind11::array_t<float> vol_start, float voxel, float miu, pybind11::array_t<float> intrinsic,
pybind11::array_t<uint16_t> depth, pybind11::array_t<uint8_t> color, pybind11::array_t<float> extrinsic2init, int width, int height)
{
  tsdf((float*)tsdf_diff.request().ptr,
  (int*)tsdf_color.request().ptr,
  (int*)tsdf_wt.request().ptr,
  vol_dim,
  (float*)vol_start.request().ptr,
  voxel,
  miu,
  (float*)intrinsic.request().ptr,
  (uint16_t*)depth.request().ptr,
  (uint8_t*)color.request().ptr,
  (float*)extrinsic2init.request().ptr,
  width,
  height);
}

PYBIND11_MODULE(tsdf_cuda, m)
{
  m.def("tsdf_update", tsdf_update);
}