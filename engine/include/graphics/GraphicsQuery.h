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

		GLuint mQueryId;
		float mTotalElapsedTime;
	};
}

#endif