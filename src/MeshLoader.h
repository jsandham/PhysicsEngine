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
			static bool load_gmesh(const std::string& filepath, std::vector<float>& vertices, std::vector<int>& connect, std::vector<int>& bconnect, std::vector<int>& groups);
	};
}

#endif