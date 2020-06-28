#ifndef __INTERNAL_MESHES_H__
#define __INTERNAL_MESHES_H__

#include <vector>

namespace PhysicsEngine
{
	class InternalMeshes
	{
	public:
		static const std::vector<float> sphereVertices;
		static const std::vector<float> sphereNormals;
		static const std::vector<float> sphereTexCoords;
		static const std::vector<int> sphereSubMeshStartIndicies;
	};
}

#endif