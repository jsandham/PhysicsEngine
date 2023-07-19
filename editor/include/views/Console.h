#ifndef CONSOLE_H__
#define CONSOLE_H__

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class Console
	{
	private:
		bool mOpen;

	public:
		Console();
		~Console();
		Console(const Console& other) = delete;
		Console& operator=(const Console& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);
	};
} // namespace PhysicsEditor

#endif
