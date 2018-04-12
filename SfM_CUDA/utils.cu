#include "utils.cuh"
#include <sstream>

// parse camera position to projection matrix
cv::Mat parse_extrinsic(const std::vector<double>& list) {
	cv::Vec3d axis{ list[3], list[4], list[5] };
	auto axis_norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
	double theta = 2 * atan2(axis_norm, list[6]);
	axis = axis / axis_norm;

	cv::Mat rotation;
	auto rod = theta * axis;
	cv::Rodrigues(rod, rotation);

	cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);
	rotation.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));
	cv::Mat translation(3, 1, CV_64F, (void*)list.data());
	translation.copyTo(extrinsic(cv::Rect(3, 0, 1, 3)));
	extrinsic.convertTo(extrinsic, CV_32F);
	return extrinsic.inv();
}

cv::Mat inv_extrinsic(const cv::Mat& extrinsic) {
	cv::Mat rotation = extrinsic(cv::Rect(0, 0, 3, 3));
	cv::Mat translation = extrinsic(cv::Rect(3, 0, 1, 3));
	cv::Mat result(3, 4, CV_64F);
	cv::Mat rotation_inv = rotation.inv();
	cv::Mat t_prime = -rotation_inv * translation;
	rotation_inv.copyTo(result(cv::Rect(0, 0, 3, 3)));
	t_prime.copyTo(result(cv::Rect(3, 0, 1, 3)));
	std::cout << result << std::endl;
	return result;
}

cv::Mat mult_extrinsic(const cv::Mat& extrinsic1, const cv::Mat& extrinsic2) {
	cv::Mat result(3, 4, CV_64F);
	result(cv::Rect(0, 0, 3, 3)) = extrinsic1(cv::Rect(0, 0, 3, 3)) * extrinsic2(cv::Rect(0, 0, 3, 3));
	result(cv::Rect(3, 0, 1, 3)) = extrinsic1(cv::Rect(0, 0, 3, 3)) * extrinsic2(cv::Rect(3, 0, 1, 3)) + extrinsic1(cv::Rect(3, 0, 1, 3));
	return result;
}

cv::Mat pack_tsdf_color(float* tsdf_ptr, uint8_t* color_ptr) {
	cv::Mat color(4096, 4096, CV_8UC3, color_ptr);
	cv::Mat tsdf(4096, 4096, CV_32FC1, tsdf_ptr);
	cv::Mat result(color.rows, color.cols, CV_32FC4, cv::Scalar(0));
	cv::Mat color_normed;
	color.convertTo(color_normed, CV_32FC3, 1. / 255.);
	cv::Mat colors[4];
	cv::split(color_normed, colors);
	colors[3] = tsdf;
	cv::merge(colors, 4, result);
	/*cv::Mat test[4];
	cv::split(result, test);*/
	//cv::imshow("test", color_normed);
	//cv::waitKey(0);
	return result;
}

std::map<double, std::vector<double>> read_trajactory(std::string filename) {
	std::map<double, std::vector<double>> result;
	std::string line;
	std::ifstream infile(filename);
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		double ts, tx, ty, tz, qx, qy, qz, qw;
		if (!(iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
		std::vector<double> pos = { tx, ty, tz, qx, qy, qz, qw };
		result.insert(std::make_pair(std::fmod(ts, 1e5), pos));
	}
	return result;
}

float mean_depth(const cv::Mat& depth) {
	int cnt = depth.rows * depth.cols;
	auto ptr = (uint16_t*)depth.data;
	double sum = 0;
	int total = 0;
	for (int i = 0; i < cnt; i++) {
		if (ptr[i] == 0)
		{
			continue;
		}
		sum += ptr[i] / 5000.;
		total++;
	}
	return static_cast<float>(sum / total);
}