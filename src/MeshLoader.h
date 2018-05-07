#ifndef __MESHLOADER_H__
#define __MESHLOADER_H__

#include <string>
#include <vector>

namespace PhysicsEngine
{
	class MeshLoader
	{
		public:
			static bool load(const std::string& filepath, std::vector<float>& vertices, std::vector<float>& normals, std::vector<float>& texCoords);
	};
}

#endif