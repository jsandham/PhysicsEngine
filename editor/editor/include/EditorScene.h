#ifndef __EDITOR_SCENE_H__
#define __EDITOR_SCENE_H__

#include <string>

#include "core/Guid.h"

namespace PhysicsEditor
{
	struct EditorScene
	{
		std::string name;
		std::string path;
		std::string metaPath;
		std::string libraryPath;
		PhysicsEngine::Guid sceneId;
		bool isDirty;
	};
}

#endif
