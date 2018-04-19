#pragma once
class Configuration {
public:
	static float prior_mrcnn_err_rate;
	static float duplicate_thresh;
};

float Configuration::prior_mrcnn_err_rate = 0.05f;
float Configuration::duplicate_thresh = 0.5f;
