#ifndef __GRAPHICSQUERY_H__
#define __GRAPHICSQUERY_H__

#include <GL/glew.h>
#include <gl/gl.h>

namespace PhysicsEngine
{
	struct GraphicsQuery
	{
		unsigned int mNumBatchDrawCalls;
		unsigned int mNumDrawCalls;
		unsigned int mVerts;
		unsigned int mTris;
		unsigned int mLines;
		unsigned int mPoints;

		unsigned int mQueryBack;
		unsigned int mQueryFront;
		GLuint mQueryId[2];
		float mTotalElapsedTime;

		GraphicsQuery::GraphicsQuery()
		{
			mNumBatchDrawCalls = 0;
			mNumDrawCalls = 0;
			mVerts = 0;
			mTris = 0;
			mLines = 0;
			mPoints = 0;

			mQueryBack = 0;
			mQueryFront = 0;
			mQueryId[0] = 0;
			mQueryId[1] = 0;
			mTotalElapsedTime = 0.0f;
		}
	};
}

#endif