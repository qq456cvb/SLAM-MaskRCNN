#include <iostream>
#include <armadillo>
#include "TSDF.h"
#include "utils.h"
#include <vector>
#include <opencv2\opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ViewShader.h"
#include <math.h>

using namespace std;
using namespace arma;


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


void filter_gaussian(cv::Mat& img, double& mean) {
	auto ptr = (uint16_t *)img.data;
	// can be performed multiple times to increase performance
	// maximum likelihood estimation
	mean = 0;
	double stddev = 0;
	int cnt = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			auto idx = i * img.cols + j;
			if (ptr[idx] > 0) {
				cnt++;
				mean += ((double)ptr[idx] - mean) / cnt;
			}
		}
	}
	cnt = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			auto idx = i * img.cols + j;
			if (ptr[idx] > 0) {
				cnt++;
				stddev += (pow((double)ptr[idx] - mean, 2) - stddev) / cnt;
			}
		}
	}
	stddev = sqrt(stddev);
	int threshold = 3;
	double mean_tmp = mean;
	mean = 0;
	cnt = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			auto idx = i * img.cols + j;
			if (ptr[idx] > 0) {
				if (abs(ptr[idx] - mean_tmp) > threshold * stddev)
				{
					ptr[idx] = 0;
				} else {
					cnt++;
					mean += ((double)ptr[idx] - mean) / cnt;
				}
			}
		}
	}
}


void show_tsdf(const TSDF& tsdf) {
	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(640, 480, "Hello Triangle", NULL, NULL);
	if (!window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);

	int height, width;
	glfwGetFramebufferSize(window, &width, &height);

	glewExperimental = GL_TRUE;
	glewInit();

	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);

	auto view_shader = std::make_shared<ViewShader>();
	auto intrinsics = tsdf.get_intrinsics();
	cv::Mat intrinsics44 = cv::Mat::eye(4, 4, CV_32F);
	intrinsics.copyTo(intrinsics44(cv::Rect(0, 0, 3, 3)));
	cv::Mat w2s_inv = intrinsics44.inv();
	cv::Matx44f w2s_invx((float*)w2s_inv.ptr());
	view_shader->setS2WMatrix(w2s_invx);
	float center[3] = { 0, 0, 0 };
	view_shader->setCameraCenter(center);
	view_shader->setVolDim(tsdf.get_dim().val[0]);
	auto vol_start_ = tsdf.get_vol_start();
	auto vol_end_ = tsdf.get_vol_end();
	float vol_start[3] = { vol_start_[0], vol_start_[1], vol_start_[2] };
	float vol_end[3] = { vol_end_[0], vol_end_[1], vol_end_[2] };
	view_shader->setVolStart(vol_start);
	view_shader->setVolEnd(vol_end);

	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	auto color = tsdf.get_tsdf_color();
	auto depth = tsdf.get_tsdf();
	auto merged = pack_tsdf_color(depth, color);
	std::cout << type2str(merged.type()) << std::endl;

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, merged.cols, merged.rows, 0, GL_RGBA, GL_FLOAT, merged.data);
	glBindTexture(GL_TEXTURE_2D, 0);

	float angle = 0.f;
	while (!glfwWindowShouldClose(window)) {
		angle += 0.01f;
		float rot[16] = { std::cosf(angle), 0, -std::sinf(angle), 0.8 * std::sinf(angle), 0, 1, 0, 0, std::sinf(angle), 0, std::cosf(angle), 0.8 - 0.8 * std::cosf(angle), 0, 0, 0, 1 };
		cv::Mat extrinsic(4, 4, CV_32F, rot);
		extrinsic = extrinsic.inv();
		cv::Mat w2s = intrinsics44 * extrinsic;
		cv::Mat s2w = w2s.inv();
		cv::Matx44f s2wx((float*)s2w.ptr());
		view_shader->setS2WMatrix(s2wx);
		center[0] = 0.8 * std::sinf(angle);
		center[2] = 0.8 - 0.8 * std::cosf(angle);
		view_shader->setCameraCenter(center);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, width, height);
		glUseProgram(view_shader->mProgram.getId());

		glBindVertexArray(view_shader->vao);

		glBindTexture(GL_TEXTURE_2D, tex);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindTexture(GL_TEXTURE_2D, 0);
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	glfwTerminate();
	return;
}

int main()
{
	cv::Scalar intrinsics{ 520.9, 521.0, 325.1, 249.7 };
	auto tsdf = std::make_shared<TSDF>(intrinsics);
	cv::String rgb_path("D://rgb-datasets//cokecan//rgb//*.png"); //select only png
	cv::String depth_path("D://rgb-datasets//cokecan//depth//*.png");
	cv::String mask_path("D://rgb-datasets//cokecan//gray_mask//*.png");
	auto traj = read_trajactory("D://rgb-datasets//cokecan//groundtruth.txt");
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
	for (size_t i = 0; i < 5; i++) {
		while (depth_timestamps[i] < mask_timestamps[j]) i++;
		auto depth_img = cv::imread(depth_fn[i], CV_LOAD_IMAGE_ANYDEPTH);
		auto mask_img = cv::imread(mask_fn[j], CV_LOAD_IMAGE_GRAYSCALE);
		auto rgb_img = cv::imread(rgb_fn[j]);
		uint16_t *depth_ptr = (uint16_t *)depth_img.data;
		uint8_t *mask_ptr = (uint8_t *)mask_img.data;
		cv::Vec3b *rgb_img_ptr = (cv::Vec3b *)rgb_img.data;
		for (int j = 0; j < depth_img.rows; j++)
		{
			for (int k = 0; k < depth_img.cols; k++)
			{
				auto idx = j * depth_img.cols + k;
				depth_ptr[idx] = depth_ptr[idx] * (mask_ptr[idx] > 0 ? 1 : 0);
				//cv::multiply(cv::Scalar(mask_ptr[idx] > 0 ? 1 : 0), rgb_img_ptr[idx], rgb_img_ptr[idx]);
			}
		}
		double mean;
		filter_gaussian(depth_img, mean);
		auto low = traj.lower_bound(depth_timestamps[i]);
		auto extrinsic = parse_pos(low->second);
		tsdf->parse_frame(depth_img, rgb_img, extrinsic, mean);
		
		while (mask_timestamps[j] < depth_timestamps[i]) j++;
	}
	show_tsdf(*tsdf);
	return 0;
}