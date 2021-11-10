#include <string>

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorProjectManager
	{
	public:
		static void newProject(Clipboard& clipboard);
		static void openProject(Clipboard& clipboard);
		static void saveProject(Clipboard& clipboard);
	};
}