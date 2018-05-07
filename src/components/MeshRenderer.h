#ifndef __MESHRENDERER_H__
#define __MESHRENDERER_H__

#include <string>
#include <vector>

#include "Component.h"

#include "../core/Mesh.h"

#include "../graphics/Material.h"
#include "../graphics/Buffer.h"
#include "../graphics/VertexArrayObject.h"

namespace PhysicsEngine
{
	class MeshRenderer : public Component
	{
		private:
			bool queued;
			bool visible;
			int meshFilter;
			int matFilter;

			Buffer vertexVBO;
			Buffer normalVBO;
			Buffer texCoordVBO;
			Buffer colourVBO;

			VertexArrayObject meshVAO;

		public:
			MeshRenderer();
			~MeshRenderer();

			bool isQueued();
			bool isVisible();
			int getMaterialFilter();
			int getMeshFilter();

			void setQueued(bool flag);
			void setMaterialFilter(int filter);
			void setMeshFilter(int filter);

			Buffer* getVertexVBO();
			Buffer* getNormalVBO();
			Buffer* getTexCoordVBO();
			Buffer* getColourVBO();
			VertexArrayObject* getMeshVAO();
	};
}

#endif