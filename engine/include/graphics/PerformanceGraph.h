#ifndef __PERFORMANCE_GRAPH_H__
#define __PERFORMANCE_GRAPH_H__

#include <vector>

#include "../core/Shader.h"

namespace PhysicsEngine
{
	typedef struct PerformanceGraph
	{
		float mX;
		float mY;
		float mWidth;
		float mHeight;
		float mRangeMin;
		float mRangeMax;
		float mCurrentSample;
		int mNumberOfSamples;

		std::vector<float> mSamples;

		GLuint mVAO;
		GLuint mVBO;

		Shader mShader;

		void init();
		void add(float sample);
	}PerformanceGraph;


	//typedef struct GBuffer
	//{
	//	GLenum mGBufferStatus;
	//	GLuint mHandle;
	//	GLuint mColor0; // position
	//	GLuint mColor1; // normal
	//	GLuint mColor2; // color + spec
	//	GLuint mDepth;
	//	Shader mShader;
	//}GBuffer;

	/*typedef struct Framebuffer
	{
		GLenum mFramebufferStatus;
		GLuint mHandle;
		Texture2D mColorBuffer;
		Texture2D mDepthBuffer;
	}Framebuffer;*/
}

#endif
