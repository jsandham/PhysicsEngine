#ifndef __DEBUG_WINDOW_H__
#define __DEBUG_WINDOW_H__

#include <vector>

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	class DebugWindow
	{
		private:
			float x;
			float y;
			float width;
			float height;

		public:
			GLHandle windowVAO;
			GLHandle vertexVBO;
			std::vector<float> vertices;

		public:
			DebugWindow(float x, float y, float width, float height);
			~DebugWindow();
	};
}

#endif