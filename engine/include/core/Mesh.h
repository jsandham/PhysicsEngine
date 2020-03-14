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
			GLuint vao;
			GLuint vbo[3];
			bool created;

		public:
			Mesh();
			Mesh(std::vector<char> data);
			~Mesh();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			void load(const std::string& filename);
			void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords, std::vector<int> subMeshStartIndices);

			bool isCreated() const;
			const std::vector<float>& getVertices() const;
			const std::vector<float>& getNormals() const;
			const std::vector<float>& getTexCoords() const;
			const std::vector<int>& getSubMeshStartIndices() const;
			Sphere getBoundingSphere() const;
			GLuint getNativeGraphicsVAO() const;

			void create();
			void destroy();
			void apply();
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