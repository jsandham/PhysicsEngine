#ifndef __GRAPHICSQUERY_H__
#define __GRAPHICSQUERY_H__

#include <gl/gl.h>

namespace PhysicsEngine
{
	struct GraphicsQuery
	{
		unsigned int numBatchDrawCalls;
		unsigned int numDrawCalls;
		unsigned int verts;
		unsigned int tris;
		unsigned int lines;
		unsigned int points;

		GLuint queryId;
		float totalElapsedTime;
	};
}

#endif