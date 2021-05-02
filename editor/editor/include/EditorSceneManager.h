#include <string>

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorSceneManager
	{
	public:
		static void newScene(Clipboard& clipboard);
		static void openScene(Clipboard& clipboard, const std::string& name, const std::string& path);
		static void saveScene(Clipboard& clipboard, const std::string& name, const std::string& path);
	};
}