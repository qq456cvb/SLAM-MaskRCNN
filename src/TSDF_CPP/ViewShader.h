#pragma once
#include <stdio.h>
#include <opencv2/core.hpp>
#include "ShaderProgram.h"
class ViewShader
{
public:
	ViewShader() {
		init();
	}
	~ViewShader() {
		free();
	}

	bool init();
	void free() { mProgram.free(); }

	//mvp is in normal opencv row-major order

	GLuint vbos[2];
	GLuint vao;
	ShaderProgram mProgram;
	int mUniformTex, mUniformS2W, mUniformC, mUniformVolDim, mUniformVolStart, mUniformVolEnd;
	int mAttribPosCoord;
	int mAttribTexCoord;

	void setS2WMatrix(const cv::Matx44f &mat);
	void setCameraCenter(float *origin);
	void setVolDim(float dim);
	void setVolStart(float *);
	void setVolEnd(float *);
protected:
};