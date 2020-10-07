//#include <algorithm>
//
//#include "../../include/graphics/PerformanceGraph.h"
//#include "../../include/core/InternalShaders.h"
//
// using namespace PhysicsEngine;
//
// void PerformanceGraph::init()
//{
//	mShader.setVertexShader(InternalShaders::graphVertexShader);
//	mShader.setFragmentShader(InternalShaders::graphFragmentShader);
//	mShader.compile();
//
//	mX = fmin(fmax(mX, 0.0f), 1.0f);
//	mY = fmin(fmax(mY, 0.0f), 1.0f);
//	mWidth = fmin(fmax(mWidth, 0.0f), 1.0f);
//	mHeight = fmin(fmax(mHeight, 0.0f), 1.0f);
//	mRangeMin = fmin(mRangeMin, mRangeMax);
//	mRangeMax = fmax(mRangeMin, mRangeMax);
//	mCurrentSample = 0.0f;
//	mNumberOfSamples = std::max(2, mNumberOfSamples);
//
//	mSamples.resize(18 * mNumberOfSamples - 18);
//
//	glGenVertexArrays(1, &mVAO);
//	glBindVertexArray(mVAO);
//
//	glGenBuffers(1, &mVBO);
//	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
//	glBufferData(GL_ARRAY_BUFFER, mSamples.size() * sizeof(float), &(mSamples[0]), GL_DYNAMIC_DRAW);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
//
//	glBindVertexArray(0);
//}
//
// void PerformanceGraph::add(float sample)
//{
//	float oldSample = mCurrentSample;
//	mCurrentSample = fmin(fmax(sample, mRangeMin), mRangeMax);
//
//	float dx = mWidth / (mNumberOfSamples - 1);
//	for (int i = 0; i < mNumberOfSamples - 2; i++) {
//		mSamples[18 * i] = mSamples[18 * (i + 1)] - dx;
//		mSamples[18 * i + 1] = mSamples[18 * (i + 1) + 1];
//		mSamples[18 * i + 2] = mSamples[18 * (i + 1) + 2];
//
//		mSamples[18 * i + 3] = mSamples[18 * (i + 1) + 3] - dx;
//		mSamples[18 * i + 4] = mSamples[18 * (i + 1) + 4];
//		mSamples[18 * i + 5] = mSamples[18 * (i + 1) + 5];
//
//		mSamples[18 * i + 6] = mSamples[18 * (i + 1) + 6] - dx;
//		mSamples[18 * i + 7] = mSamples[18 * (i + 1) + 7];
//		mSamples[18 * i + 8] = mSamples[18 * (i + 1) + 8];
//
//		mSamples[18 * i + 9] = mSamples[18 * (i + 1) + 9] - dx;
//		mSamples[18 * i + 10] = mSamples[18 * (i + 1) + 10];
//		mSamples[18 * i + 11] = mSamples[18 * (i + 1) + 11];
//
//		mSamples[18 * i + 12] = mSamples[18 * (i + 1) + 12] - dx;
//		mSamples[18 * i + 13] = mSamples[18 * (i + 1) + 13];
//		mSamples[18 * i + 14] = mSamples[18 * (i + 1) + 14];
//
//		mSamples[18 * i + 15] = mSamples[18 * (i + 1) + 15] - dx;
//		mSamples[18 * i + 16] = mSamples[18 * (i + 1) + 16];
//		mSamples[18 * i + 17] = mSamples[18 * (i + 1) + 17];
//	}
//
//	float dz1 = 1.0f - (mCurrentSample - mRangeMin) / (mRangeMax - mRangeMin);
//	float dz2 = 1.0f - (oldSample - mRangeMin) / (mRangeMax - mRangeMin);
//
//	float x_ndc = 2.0f * mX - 1.0f;
//	float y0_ndc = 1.0f - 2.0f * (mY + mHeight);
//	float y1_ndc = 1.0f - 2.0f * (mY + mHeight * dz1);
//	float y2_ndc = 1.0f - 2.0f * (mY + mHeight * dz2);
//
//	mSamples[18 * (mNumberOfSamples - 2)] = x_ndc + dx * (mNumberOfSamples - 2);
//	mSamples[18 * (mNumberOfSamples - 2) + 1] = y2_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 2] = 0.0f;
//
//	mSamples[18 * (mNumberOfSamples - 2) + 3] = x_ndc + dx * (mNumberOfSamples - 2);
//	mSamples[18 * (mNumberOfSamples - 2) + 4] = y0_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 5] = 0.0f;
//
//	mSamples[18 * (mNumberOfSamples - 2) + 6] = x_ndc + dx * (mNumberOfSamples - 1);
//	mSamples[18 * (mNumberOfSamples - 2) + 7] = y0_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 8] = 0.0f;
//
//	mSamples[18 * (mNumberOfSamples - 2) + 9] = x_ndc + dx * (mNumberOfSamples - 2);
//	mSamples[18 * (mNumberOfSamples - 2) + 10] = y2_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 11] = 0.0f;
//
//	mSamples[18 * (mNumberOfSamples - 2) + 12] = x_ndc + dx * (mNumberOfSamples - 1);
//	mSamples[18 * (mNumberOfSamples - 2) + 13] = y0_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 14] = 0.0f;
//
//	mSamples[18 * (mNumberOfSamples - 2) + 15] = x_ndc + dx * (mNumberOfSamples - 1);
//	mSamples[18 * (mNumberOfSamples - 2) + 16] = y1_ndc;
//	mSamples[18 * (mNumberOfSamples - 2) + 17] = 0.0f;
//
//	glBindVertexArray(mVAO);
//	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
//
//	glBufferSubData(GL_ARRAY_BUFFER, 0, mSamples.size() * sizeof(float), &(mSamples[0]));
//
//	glBindVertexArray(0);
//}