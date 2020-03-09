#ifndef __INTERNAL_MESHES_H__
#define __INTERNAL_MESHES_H__

#include <vector>

namespace PhysicsEngine
{
	class InternalMeshes
	{
	public:
		static std::vector<float> sphereVertices;
		static std::vector<float> sphereNormals;
		static std::vector<float> sphereTexCoords;
		static std::vector<int> sphereSubMeshStartIndicies;
		
	};
}

#endif