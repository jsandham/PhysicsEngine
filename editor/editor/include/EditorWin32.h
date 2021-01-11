#ifndef __EDITORWIN32_H__
#define __EDITORWIN32_H__

#include <windows.h>

#include "Editor.h"

namespace PhysicsEditor
{
	class EditorWin32
	{
    private:
        Editor mEditor;

    public:
        EditorWin32();
        ~EditorWin32();
        EditorWin32(const EditorWin32& other) = delete;
        EditorWin32& operator=(const EditorWin32& other) = delete;

        void init(HWND window, int width, int height);
        void cleanUp();
        void update(HWND window, bool editorApplicationActive);

        bool isQuitCalled() const;
	};
}

#endif