#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

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

		public:
			Mesh();
			~Mesh();
	};
}

#endif