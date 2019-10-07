#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

#include "Guid.h"
#include "Asset.h"
#include "Sphere.h"

#include "../graphics/GraphicsHandle.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshHeader
	{
		Guid meshId;
		size_t verticesSize;
		size_t normalsSize;
		size_t texCoordsSize;
		size_t subMeshVertexStartIndiciesSize;
	};
#pragma pack(pop)

	class Mesh : public Asset
	{
		public:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;
			std::vector<int> subMeshVertexStartIndices;

			GraphicsHandle vao;
			GraphicsHandle vbo[3];

			bool isCreated;

		public:
			Mesh();
			Mesh(std::vector<char> data);
			~Mesh();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			Sphere getBoundingSphere() const;
	};

	template <>
	const int AssetType<Mesh>::type = 5;
}

#endif