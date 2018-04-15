#include <iostream>
#include "tsdf.cuh"
#include "utils.cuh"
#include <vector>
#include <opencv2\opencv.hpp>
#include <math.h>
#include <memory>
#include <cuda_runtime.h>
#include "viewer.cuh"


using namespace std;


string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//void show_tsdf(const TSDF& tsdf) {
//	if (!glfwInit()) {
//		fprintf(stderr, "ERROR: could not start GLFW3\n");
//		return;
//	}
//
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
//	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//	GLFWwindow* window = glfwCreateWindow(1024, 768, "Hello Triangle", NULL, NULL);
//	if (!window) {
//		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
//		glfwTerminate();
//		return;
//	}
//	glfwMakeContextCurrent(window);
//
//	int height, width;
//	glfwGetFramebufferSize(window, &width, &height);
//
//	glewExperimental = GL_TRUE;
//	glewInit();
//
//	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
//	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
//	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
//
//	auto view_shader = std::make_shared<ViewShader>();
//	view_shader->setVolDim(tsdf.get_dim().val[0]);
//	auto vol_start_ = tsdf.get_vol_start();
//	auto vol_end_ = tsdf.get_vol_end();
//	float vol_start[3] = { vol_start_[0], vol_start_[1], vol_start_[2] };
//	float vol_end[3] = { vol_end_[0], vol_end_[1], vol_end_[2] };
//	view_shader->setVolStart(vol_start);
//	view_shader->setVolEnd(vol_end);
//
//	GLuint tex;
//	glGenTextures(1, &tex);
//	glBindTexture(GL_TEXTURE_2D, tex);
//
//	auto color = tsdf.get_tsdf_color();
//	auto depth = tsdf.get_tsdf_diff();
//	auto merged = pack_tsdf_color(depth, color);
//	std::cout << type2str(merged.type()) << std::endl;
//
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, merged.cols, merged.rows, 0, GL_RGBA, GL_FLOAT, merged.data);
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	float angle = 0.f;
//	float dist = tsdf.mean_depth_;
//	while (!glfwWindowShouldClose(window)) {
//		angle += 0.01f;
//		float rot[16] = { std::cosf(angle), 0, -std::sinf(angle), dist * std::sinf(angle), 0, 1, 0, 0, std::sinf(angle), 0, std::cosf(angle), dist - dist * std::cosf(angle), 0, 0, 0, 1 };
//		cv::Mat extrinsic(4, 4, CV_32F, rot);
//		cv::Mat s2w = extrinsic * tsdf.get_intrinsic_inv();
//		cv::Matx44f s2wx((float*)s2w.ptr());
//		view_shader->setS2WMatrix(s2wx);
//		float center[3] = { 0 };
//		center[0] = (dist + 0.5f) * std::sinf(angle);
//		center[2] = (dist + 0.5f) - (dist + 0.5f) * std::cosf(angle);
//		view_shader->setCameraCenter(center);
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//		glViewport(0, 0, width, height);
//		glUseProgram(view_shader->mProgram.getId());
//
//		glBindVertexArray(view_shader->vao);
//
//		glBindTexture(GL_TEXTURE_2D, tex);
//		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
//		glBindTexture(GL_TEXTURE_2D, 0);
//		glfwPollEvents();
//		glfwSwapBuffers(window);
//	}
//	glfwTerminate();
//	return;
//}

int main()
{
	cv::Scalar intrinsics{ 520.9f, 521.0f, 325.1f, 249.7f };
	auto tsdf = std::make_shared<TSDF>(intrinsics);
	cv::String rgb_path("D://rgb-datasets//desk//rgb//*.png"); //select only png
	cv::String depth_path("D://rgb-datasets//desk//depth//*.png");
	cv::String mask_path("D://rgb-datasets//desk//mask//*.png");
	auto traj = read_trajactory("D://rgb-datasets//desk//groundtruth.txt");
	vector<cv::String> rgb_fn, depth_fn, mask_fn;
	cv::glob(rgb_path, rgb_fn, false);
	cv::glob(depth_path, depth_fn, false);
	cv::glob(mask_path, mask_fn, false);
	size_t j = 0;
	vector<double> depth_timestamps, mask_timestamps;
	transform(depth_fn.begin(), depth_fn.end(), back_inserter(depth_timestamps),
		[](const cv::String& fn) -> float {
		return stod(fn.substr(fn.find_last_of("/") + 6, fn.find_last_of(".") - fn.find_last_of("/") - 6));
	});
	transform(mask_fn.begin(), mask_fn.end(), back_inserter(mask_timestamps),
		[](const cv::String& fn) -> float {
		return stod(fn.substr(fn.find_last_of("/") + 6, fn.find_last_of(".") - fn.find_last_of("/") - 6));
	});

	double begin = 68164;
	double end = 68170;
	Viewer *viewer = nullptr;
	for (size_t i = 0; i < 30; i++) {
		if (i >= depth_timestamps.size()) break;
		if (depth_timestamps[i] < begin || depth_timestamps[i] > end) continue;
		while (depth_timestamps[i] < mask_timestamps[j]) i++;
		while (mask_timestamps[j] < depth_timestamps[i]) j++;
		auto depth_img = cv::imread(depth_fn[i], CV_LOAD_IMAGE_ANYDEPTH);
		auto mask_img = cv::imread(mask_fn[j], CV_LOAD_IMAGE_GRAYSCALE);
		auto rgb_img = cv::imread(rgb_fn[j]);
		std::cout << "processing: " << i << std::endl;
		if (!viewer)
		{
			viewer = new Viewer(depth_img.cols, depth_img.rows);
		}
		//cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);

		/*cv::Mat obj_img(mask_img.rows, mask_img.cols, CV_8UC1, cv::Scalar(0));
		auto mask_ptr = mask_img.data;
		for (int k = 0; k < mask_img.rows * mask_img.cols; k++)
		{
			if (mask_ptr[k] == 9)
			{
				obj_img.data[k] = 255;
			}
		}
		cv::imshow("mask", obj_img);
		cv::waitKey();*/

		auto mean = mean_depth(depth_img);

		auto low = traj.lower_bound(depth_timestamps[i]);
		auto extrinsic = parse_extrinsic(low->second);
		tsdf->parse_frame(depth_img, rgb_img, mask_img, extrinsic, mean);
	}
	if (viewer) {
		float angle = 0.f;
		while (1) {
			angle += 0.01f;
			viewer->show_tsdf(*tsdf, angle, tsdf->mean_depth_);
		}
	}
	
	delete viewer;
	return 0;
}