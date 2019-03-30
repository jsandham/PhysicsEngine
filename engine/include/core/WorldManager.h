#ifndef __WORLDMANAGER_H__
#define __WORLDMANAGER_H__

#include <string>

#include "World.h"
#include "Input.h"

namespace PhysicsEngine
{
	class WorldManager
	{
		private:
			Scene scene;
			AssetBundle bundle;

			World* world;

		public:
			WorldManager(Scene scene, AssetBundle bundle);
			~WorldManager();

			void init();
			bool update(Input input);
	};
}

#endif