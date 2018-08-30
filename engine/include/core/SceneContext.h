#ifndef __SCENECONTEXT_H__
#define __SCENECONTEXT_H__

#include <string>

#include "Manager.h"

namespace PhysicsEngine
{
	class SceneContext
	{
		private:
			int sceneToLoadIndex;
			std::string sceneToLoad;

			std::vector<Scene> scenes;

		public:
			SceneContext();
			~SceneContext();

			void add(Scene scene);
			void setSceneToLoad(std::string sceneName);
			void setSceneToLoadIndex(int index);
			int getSceneToLoadIndex();

	};
}

#endif