#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

#include "Guid.h"
#include "Asset.h"
#include "Sphere.h"

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshHeader
	{
		Guid meshId;
		size_t verticesSize;
		size_t normalsSize;
		size_t texCoordsSize;
		size_t subMeshStartIndiciesSize;
	};
#pragma pack(pop)

	class Mesh : public Asset
	{
		public:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;
			std::vector<int> subMeshStartIndicies;

		public:
			Mesh();
			Mesh(std::vector<char> data);
			~Mesh();

			Sphere getBoundingSphere() const;
	};
}

#endif