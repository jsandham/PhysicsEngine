#ifndef __ASSET_LOADER_H__
#define __ASSET_LOADER_H__

#include <string>
#include <vector>

#include "Shader.h"
#include "Texture2D.h"
#include "Mesh.h"
#include "GMesh.h"
#include "Font.h"

namespace PhysicsEngine
{
	class AssetLoader
	{
		public:
			static bool load(const std::string& filepath, Shader& shader);
			// right now I load textures with stb_load which can take png etc. Should I instead only load ktx file format textures in the engine and have png converted to this format before passing to the engine???
			static bool load(const std::string& filepath, Texture2D& texture); 
			static bool load(const std::string& filepath, Mesh& mesh);
			//static bool load2(const std::string& filepath, Mesh& mesh);

			static bool load(const std::string& filepath, GMesh& gmesh);
			//static bool load(const std::string& filepath, Font& font);

		private:
			static void split(const std::string &s, char delim, std::vector<std::string> &elems);
			static std::vector<std::string> split(const std::string &s, char delim);
	};
}

#endif