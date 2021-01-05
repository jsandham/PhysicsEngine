#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class Window //View?
	{
	protected:
		bool windowActive;

	public:
		Window();
		virtual ~Window() = 0;

		virtual void init(EditorClipboard& clipboard);
		virtual void update(EditorClipboard& clipboard, bool isOpenedThisFrame);
	};
}



//void render(EditorClipboard& clipboard, bool isOpenedThisFrame);
//void render(EditorClipboard& clipboard, bool isOpenedThisFrame);
//void render(EditorClipboard& clipboard, bool isOpenedThisFrame);
//void render(EditorClipboard& clipboard, bool isOpenedThisFrame);
//void render(EditorClipboard& clipboard, bool editorBecameActiveThisFrame, bool isOpenedThisFrame);

#endif