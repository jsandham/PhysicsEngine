#ifndef __GRAPHICSQUERY_H__
#define __GRAPHICSQUERY_H__

#include <gl/gl.h>

namespace PhysicsEngine
{
	struct GraphicsQuery
	{
		unsigned int numBatchDrawCalls;
		unsigned int numDrawCalls;

		GLuint queryId;
		GLuint64 totalElapsedTime;
	};
}

#endif