#ifndef __LINERENDERER_H__
#define __LINERENDERER_H__

#include <vector>

#include "Component.h"

#include "../graphics/Buffer.h"
#include "../graphics/VertexArrayObject.h"

#include "../entities/Entity.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class LineRenderer : public Component
	{
		bool queued;
		int matFilter;
		float lineWidth;
		std::vector<float> vertices;

		Buffer vertexVBO;
		Buffer colourVBO;

		VertexArrayObject lineVAO;

		public:
			LineRenderer();
			~LineRenderer();

			void initLineData();
			void updateLineData();
			void draw();

			void setQueued(bool flag);
			void setMaterialFilter(int filter);
			void setLineWidth(float width);
			void setVertices(std::vector<float> vertices);

			bool isQueued();
			int getMaterialFilter();
			float getLineWidth();
			std::vector<float> getVertices();
	};
}

#endif