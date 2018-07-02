#ifndef __MESHLOADER_H__
#define __MESHLOADER_H__

#include <string>
#include <vector>

#include "core/GMesh.h"
#include "core/Mesh.h"

namespace PhysicsEngine
{
	class MeshLoader
	{
		public:
			static bool load(const std::string& filepath, Mesh& mesh);
			static bool load_gmesh(const std::string& filepath, GMesh& gmesh);
	};
}

#endif