#include "../../include/core/SceneContext.h"

using namespace PhysicsEngine;

SceneContext::SceneContext()
{
	sceneToLoadIndex = -1;
}

SceneContext::~SceneContext()
{

}

void SceneContext::add(Scene scene)
{
	sceneToLoadIndex = 0;

	scenes.push_back(scene);
}

void SceneContext::setSceneToLoad(std::string sceneName)
{
	bool sceneFound = false;
	for(unsigned int i = 0; i < scenes.size(); i++){
		std::cout << "scene name: " << scenes[i].name << std::endl;
		if(scenes[i].name == sceneName){
			sceneToLoadIndex = i;
			sceneToLoad = sceneName;
			sceneFound = true;
			break;
		}
	}

	if(!sceneFound){
		std::cout << "Error: Scene name " << sceneName << " not found" << std::endl;
	}
}

void SceneContext::setSceneToLoadIndex(int index)
{
	if(index >= scenes.size()){
		std::cout << "Error: scene to load index out of range" << std::endl;
		return;
	}

	sceneToLoadIndex = index;
}

int SceneContext::getSceneToLoadIndex()
{
	return sceneToLoadIndex;
}