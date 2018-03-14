#include "utils.h"
#include <sstream>

// parse camera position to projection matrix
cv::Mat parse_pos(const std::vector<double>& list) {
	double theta = acos(list[6]);
	cv::Vec3d axis{ list[3], list[4], list[5] };
	axis /= sin(theta);

	cv::Mat rotation;
	auto rod = 2 * theta * axis;
	cv::Rodrigues(rod, rotation);
	rotation = rotation.inv();

	cv::Mat extrinsic(3, 4, CV_64F, cv::Scalar(0));
	rotation.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));
	cv::Mat translation(3, 1, CV_64F, (void*)list.data());
	std::cout << translation << std::endl;
	cv::Mat Rc = -rotation * translation;
	Rc.copyTo(extrinsic(cv::Rect(3, 0, 1, 3)));
	return extrinsic;
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

cv::Mat pack_tsdf_color(const cv::Mat& tsdf, const cv::Mat& color) {
	cv::Mat result(color.rows, color.cols, CV_32FC4, cv::Scalar(0));
	cv::Mat color_normed;
	color.convertTo(color_normed, CV_32FC3, 1. / 255.);
	cv::Mat colors[4];
	cv::split(color_normed, colors);
	colors[3] = tsdf;
	cv::merge(colors, 4, result);
	cv::Mat test[4];
	cv::split(result, test);
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