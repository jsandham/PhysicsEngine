#ifndef __ASSET_LOADER_H__
#define __ASSET_LOADER_H__

#include <string>
#include <vector>

#include "Shader.h"
#include "Texture2D.h"
#include "Mesh.h"
#include "GMesh.h"

namespace PhysicsEngine
{
	class AssetLoader
	{
		public:
			static bool load(const std::string& filepath, Shader& shader);
			static bool load(const std::string& filepath, Texture2D& texture); 
			static bool load(const std::string& filepath, Mesh& mesh);
			static bool load(const std::string& filepath, GMesh& gmesh);
	};
}

#endif