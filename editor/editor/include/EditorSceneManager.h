#include <string>
#include <filesystem>

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorSceneManager
	{
	public:
		static void newScene(Clipboard& clipboard);
		static void openScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path);
		static void saveScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path);
	};
}