#ifndef EDITOR_SCENE_MANAGER_H__
#define EDITOR_SCENE_MANAGER_H__

#include <string>
#include <filesystem>

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorSceneManager
	{
	public:
		static void newScene(Clipboard& clipboard, const std::string& sceneName);
		static void openScene(Clipboard& clipboard, const std::filesystem::path& scenePath);
		static void saveScene(Clipboard& clipboard, const std::filesystem::path& scenePath);

		static void populateScene(Clipboard& clipboard);
	};
}

#endif