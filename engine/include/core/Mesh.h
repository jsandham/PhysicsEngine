#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

#include "Guid.h"
#include "Asset.h"

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshHeader
	{
		Guid meshId;
		size_t verticesSize;
		size_t normalsSize;
		size_t texCoordsSize;
	};
#pragma pack(pop)

	class Mesh : public Asset
	{
		public:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;

			GLHandle meshVAO;
			GLHandle vertexVBO;
			GLHandle normalVBO;
			GLHandle texCoordVBO;

		public:
			Mesh();
			Mesh(unsigned char* data);
			~Mesh();

			void* operator new(size_t size);
			void operator delete(void*);

			void apply();
	};
}

#endif