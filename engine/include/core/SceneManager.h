#ifndef __SCENEMANAGER_H__
#define __SCENEMANAGER_H__

#include <string>

#include "SceneContext.h"
#include "Manager.h"

namespace PhysicsEngine
{
	class SceneManager
	{
		private:
			int loadingSceneIndex;
			int activeSceneIndex;

			SceneContext context;
			Scene* activeScene;
			Scene* loadingScene; 

			std::vector<Scene> scenes;
			std::vector<AssetFile> assetFiles;

			Manager* manager;

		public:
			SceneManager();
			~SceneManager();

			void add(Scene scene);
			void add(AssetFile assetFile);

			void init();
			bool update();
	};
}

#endif