#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	class Mesh
	{
		public:
			int meshId;
			int globalIndex;

			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;

			GLHandle meshVAO;
			GLHandle vertexVBO;
			GLHandle normalVBO;
			GLHandle texCoordVBO;

		public:
			Mesh();
			~Mesh();

			void apply();
	};
}

#endif