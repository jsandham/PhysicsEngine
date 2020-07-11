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
		Guid mMeshId;
		size_t mVerticesSize;
		size_t mNormalsSize;
		size_t mTexCoordsSize;
		size_t mSubMeshVertexStartIndiciesSize;
	};
#pragma pack(pop)

	class Mesh : public Asset
	{
		private:
			std::vector<float> mVertices;
			std::vector<float> mNormals;
			std::vector<float> mTexCoords;
			std::vector<int> mSubMeshVertexStartIndices;
			GLuint mVao;
			GLuint mVbo[3];
			bool mCreated;

		public:
			Mesh();
			Mesh(const std::vector<char>& data);
			~Mesh();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(const std::vector<char>& data);

			void load(const std::string& filename);
			void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords, std::vector<int> subMeshStartIndices);

			Sphere computeBoundingSphere() const;

			bool isCreated() const;
			const std::vector<float>& getVertices() const;
			const std::vector<float>& getNormals() const;
			const std::vector<float>& getTexCoords() const;
			const std::vector<int>& getSubMeshStartIndices() const;
			int getSubMeshStartIndex(int subMeshIndex) const;
			int getSubMeshEndIndex(int subMeshIndex) const;
			int getSubMeshCount() const;
			GLuint getNativeGraphicsVAO() const;

			void create();
			void destroy();
			void apply();
	};

	template <>
	const int AssetType<Mesh>::type = 5;

	template <typename T>
	struct IsMesh { static const bool value; };

	template <typename T>
	const bool IsMesh<T>::value = false;

	template<>
	const bool IsMesh<Mesh>::value = true;
	template<>
	const bool IsAsset<Mesh>::value = true;
	template<>
	const bool IsAssetInternal<Mesh>::value = true;
}

#endif