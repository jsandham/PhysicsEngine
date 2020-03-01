#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "Guid.h"
#include "Asset.h"
#include "Sphere.h"

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
		private:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;
			std::vector<int> subMeshVertexStartIndices;

		public:
			GLuint vao;
			GLuint vbo[3];

			bool isCreated;

		public:
			Mesh();
			Mesh(std::vector<char> data);
			~Mesh();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void load(const std::string& filename);

			const std::vector<float>& getVertices() const;
			const std::vector<float>& getNormals() const;
			const std::vector<float>& getTexCoords() const;
			const std::vector<int>& getSubMeshStartIndices() const;
			Sphere getBoundingSphere() const;
	};

	template <>
	const int AssetType<Mesh>::type = 5;

	template <typename T>
	struct IsMesh { static bool value; };

	template <typename T>
	bool IsMesh<T>::value = false;

	template<>
	bool IsMesh<Mesh>::value = true;
	template<>
	bool IsAsset<Mesh>::value = true;
}

#endif