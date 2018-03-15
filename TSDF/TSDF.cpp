#include "TSDF.h"
#include "utils.h"
#include <pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

//#define VISUALIZE


std::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}


TSDF::TSDF(cv::Scalar intrinsics)
{
	double fx = intrinsics[0];
	double fy = intrinsics[1];
	double cx = intrinsics[2];
	double cy = intrinsics[3];
	this->intrinsics_.at<double>(0, 0) = fx;
	this->intrinsics_.at<double>(1, 1) = fy;
	this->intrinsics_.at<double>(0, 2) = cx;
	this->intrinsics_.at<double>(1, 2) = cy;
	intrinsics_inv_ = intrinsics_.inv();
	std::cout << intrinsics_ << std::endl;
	std::cout << intrinsics_inv_ << std::endl;
}

TSDF::~TSDF()
{
}


// notice: we get tsdf in left-handed coordinates
void TSDF::parse_frame(const cv::Mat& depth, const cv::Mat& color, const cv::Mat& extrinsics, double mean_depth) {
	// init bounding box
	if (!init_) {
		// TODO: decide the best scale factor to transform volume into unit box
		init_ = true;
		init_pos_inv_ = inv_extrinsic(extrinsics);
		cv::Mat points;
		cv::Mat depth_mask;
		depth.convertTo(depth_mask, CV_8UC1);
		cv::findNonZero(depth_mask, points);
		cv::Rect min_rect = cv::boundingRect(points);

		// we use the diagonal as the volume side
		cv::Mat tl = intrinsics_inv_ * cv::Mat(3, 1, CV_64F, std::vector<double>({ (double)min_rect.tl().x, color.rows - 1 - (double)min_rect.tl().y, 1.0 }).data());
		cv::Mat br = intrinsics_inv_ * cv::Mat(3, 1, CV_64F, std::vector<double>({ (double)min_rect.br().x, color.rows - 1 - (double)min_rect.br().y, 1.0 }).data());
		tl = tl * mean_depth / 5000;
		br = br * mean_depth / 5000;
		std::cout << tl << std::endl;
		std::cout << br << std::endl;

		double half_side = sqrt(pow(tl.at<double>(0, 0) - br.at<double>(0, 0), 2) + pow(tl.at<double>(1, 0) - br.at<double>(1, 0), 2));
		cv::Mat center = intrinsics_inv_ * cv::Mat(3, 1, CV_64F, std::vector<double>({ ((double)min_rect.br().x + (double)min_rect.tl().x) / 2.0, 
			(color.rows - 1 - (double)min_rect.br().y + color.rows - 1 - (double)min_rect.tl().y) / 2.0, 1.0 }).data());
		center = center * mean_depth / 5000;
		cv::Vec3d center_(center);
		vol_start_ = center_ - cv::Vec3d(half_side, half_side, half_side);
		vol_end_ = center_ + cv::Vec3d(half_side, half_side, half_side);
		cv::divide(vol_end_ - vol_start_, vol_dim_, vol_res_);

		mu_ = 2 * vol_res_;
		tsdf_ = cv::Mat(tex_dim_[1], tex_dim_[0], CV_32F, cv::Scalar(0));
		tsdf_wt_ = cv::Mat(tex_dim_[1], tex_dim_[0], CV_32S, cv::Scalar(0));
		tsdf_color_ = cv::Mat(tex_dim_[1], tex_dim_[0], CV_32SC3, cv::Scalar(0));
		parse_frame(depth, color, extrinsics, mean_depth);
	}
	else {
#ifdef VISUALIZE
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		cv::Mat test(480, 640, CV_8UC1, cv::Scalar(0));
#endif
		float* tsdf_ptr = (float*)tsdf_.data;
		cv::Vec3i* tsdf_color_ptr = (cv::Vec3i*)tsdf_color_.data;
		int32_t*tsdf_wt_ptr = (int32_t*)tsdf_wt_.data;
		for (int i = 0; i < tex_dim_[1]; i++) {
			for (int j = 0; j < tex_dim_[0]; j++) {
				auto flattened_idx = i * tex_dim_[0] + j;
				auto x_idx = flattened_idx / (vol_dim_[1] * vol_dim_[2]);
				auto y_idx = flattened_idx / vol_dim_[2] - x_idx * vol_dim_[1];
				auto z_idx = flattened_idx % vol_dim_[2];

				cv::Vec3d idx(x_idx, y_idx, z_idx);
				//std::cout << idx << std::endl;
				cv::Vec3d pos_inhomo;
				cv::multiply(idx, vol_res_, pos_inhomo);
				cv::add(vol_start_, pos_inhomo, pos_inhomo);

				// assume identity camera matrix
				cv::Mat pos_homo(4, 1, CV_64F, cv::Scalar(1));
				std::memmove(pos_homo.data, pos_inhomo.val, 3 * sizeof(double));
				for (int l = 0; l < 3; l++) pos_homo.at<double>(l, 0) = pos_inhomo[l];

				cv::Mat proj = mult_extrinsic(init_pos_inv_, extrinsics) * pos_homo;
				cv::Mat pixel = intrinsics_ * proj;
				pixel /= pixel.at<double>(2, 0);
				double x = pixel.at<double>(0, 0);
				double y = pixel.at<double>(1, 0);

				if (x < 0 || x > 639 || y < 0 || y > 479) {
					continue;
				}
				// TODO: add interpolation
				// TODO: incorporate viewing angle information
				double diff = ((double)depth.at<uint16_t>(color.rows - y - 1, x)) / 5000 - proj.at<double>(2, 0);
				
				if (depth.at<uint16_t>(color.rows - y - 1, x) == 0) diff = mu_[0];
				
				diff = std::max(std::min(diff, mu_[0]), -mu_[0]);

				int weight = 1;
				tsdf_ptr[flattened_idx] = tsdf_ptr[flattened_idx] * tsdf_wt_ptr[flattened_idx] + weight * float(diff);
				tsdf_ptr[flattened_idx] /= (tsdf_wt_ptr[flattened_idx] + weight);
				// only consider valid ones
				if (depth.at<uint16_t>(color.rows - y - 1, x) != 0) {

					const auto& color_elem = color.at<cv::Vec3b>(color.rows - y - 1, x);
					cv::Vec3i colors_to_add = { color_elem[0], color_elem[1], color_elem[2] };
					colors_to_add *= weight;
					cv::multiply(tsdf_color_ptr[flattened_idx], tsdf_wt_ptr[flattened_idx], tsdf_color_ptr[flattened_idx]);
					cv::add(tsdf_color_ptr[flattened_idx], colors_to_add, tsdf_color_ptr[flattened_idx]);
					tsdf_color_ptr[flattened_idx] /= (tsdf_wt_ptr[flattened_idx] + weight);
				}

				tsdf_wt_ptr[flattened_idx] += weight;
#ifdef VISUALIZE
				// visualize
				pcl::PointXYZRGB point;
				double ratio = (diff + mu_[0]) / mu_[0] / 2;
				if (ratio > 0.5 || ratio < 0.1) continue;
				test.at<unsigned char>(y, x) = 255;
				point.x = pos_inhomo.val[0];
				point.y = pos_inhomo.val[1];
				point.z = pos_inhomo.val[2];

				point.r = tsdf_color_ptr[flattened_idx].val[2];
				point.g = tsdf_color_ptr[flattened_idx].val[1];
				point.b = tsdf_color_ptr[flattened_idx].val[0];
				point_cloud_ptr->points.push_back(point);
#endif
			}
		}
#ifdef VISUALIZE
		cv::imshow("VIS", test);
		cv::waitKey(30);
		auto viewer = rgbVis(point_cloud_ptr);
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			// std::this_thread::sleep (std::posix_time::microseconds (100000));
		}
#endif
	}
}

cv::Mat TSDF::get_tsdf_color() const {
	return tsdf_color_;
}

cv::Mat TSDF::get_tsdf() const {
	return tsdf_;
}

cv::Vec3i TSDF::get_dim() const {
	return vol_dim_;
}

cv::Vec3d TSDF::get_vol_start() const {
	return vol_start_;
}

cv::Vec3d TSDF::get_vol_end() const {
	return vol_end_;
}

cv::Mat TSDF::get_intrinsics() const {
	return intrinsics_;
}