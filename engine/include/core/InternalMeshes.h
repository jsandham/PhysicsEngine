#ifndef __INTERNAL_MESHES_H__
#define __INTERNAL_MESHES_H__

#include <vector>

#include "Mesh.h"
#include "Guid.h"

namespace PhysicsEngine
{
	class World;

	class InternalMeshes
	{
	public:
		static const std::vector<float> sphereVertices;
		static const std::vector<float> sphereNormals;
		static const std::vector<float> sphereTexCoords;
		static const std::vector<int> sphereSubMeshStartIndicies;

		static const std::vector<float> cubeVertices;
		static const std::vector<float> cubeNormals;
		static const std::vector<float> cubeTexCoords;
		static const std::vector<int> cubeSubMeshStartIndicies;

		static const Guid sphereMeshId;
		static const Guid cubeMeshId;

		static Guid loadSphereMesh(World* world);
		static Guid loadCubeMesh(World* world);

	private:
		static Guid loadInternalMesh(World* world,
									 const Guid meshId,
									 const std::vector<float>& vertices,
									 const std::vector<float>& normals,
									 const std::vector<float>& texCoords,
									 const std::vector<int>& startIndices);
	};
}

#endif