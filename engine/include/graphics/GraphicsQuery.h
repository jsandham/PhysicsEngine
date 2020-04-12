#ifndef __GRAPHICSQUERY_H__
#define __GRAPHICSQUERY_H__

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
	};
}

#endif