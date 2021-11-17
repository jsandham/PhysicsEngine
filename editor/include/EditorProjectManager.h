#include <string>
#include <filesystem>

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorProjectManager
	{
	public:
		static void newProject(Clipboard& clipboard, const std::filesystem::path& projectPath);
		static void openProject(Clipboard& clipboard, const std::filesystem::path& projectPath);
		static void saveProject(Clipboard& clipboard);
	};
}