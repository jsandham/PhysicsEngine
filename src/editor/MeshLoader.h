#ifndef __MESHLOADER_H__
#define __MESHLOADER_H__

#include <string>
#include <vector>

#include "../core/Mesh.h"
#include "../core/GMesh.h"

namespace PhysicsEngine
{
	class MeshLoader
	{
		public:
			static bool load(const std::string& filepath, Mesh& mesh);
			static bool load(const std::string& filepath, GMesh& gmesh);
	};
}

#endif