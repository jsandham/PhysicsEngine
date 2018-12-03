#ifndef __DEBUG_WINDOW_H__
#define __DEBUG_WINDOW_H__

#include <vector>

#include "../core/Texture2D.h"
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
			GLHandle texCoordVBO;

			std::vector<float> vertices;
			std::vector<float> texCoords;

		public:
			DebugWindow(float x, float y, float width, float height);
			~DebugWindow();
	};
}

#endif