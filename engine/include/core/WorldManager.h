#ifndef __WORLDMANAGER_H__
#define __WORLDMANAGER_H__

#include <string>

#include "SceneContext.h"
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



	// class WorldManager
	// {
	// 	private:
	// 		int loadingSceneIndex;
	// 		int activeSceneIndex;

	// 		SceneContext context;
	// 		Scene* activeScene;
	// 		Scene* loadingScene; 

	// 		std::vector<Scene> scenes;
	// 		std::vector<AssetFile> assetFiles;

	// 		World* world;

	// 	public:
	// 		WorldManager();
	// 		~WorldManager();

	// 		void add(Scene scene);
	// 		void add(AssetFile assetFile);

	// 		void init();
	// 		bool update(Input input);
	// };
}

#endif