#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

namespace PhysicsEngine
{
	class Mesh
	{
		public:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;
			std::vector<float> colours;

		public:
			Mesh();
			~Mesh();
	};
}

#endif