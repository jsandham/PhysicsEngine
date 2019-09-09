#ifndef __WORLDMANAGER_H__
#define __WORLDMANAGER_H__

#include <string>

#include "World.h"
#include "Input.h"
#include "Time.h"

namespace PhysicsEngine
{
	class WorldManager
	{
		private:
			World world;

		public:
			WorldManager();
			~WorldManager();

			bool load(std::string sceneFilePath, std::vector<std::string> assetFilePaths);
			void init();
			void update(Time time, Input input);
	};
}

#endif