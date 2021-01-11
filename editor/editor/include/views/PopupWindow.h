#ifndef __POPUPWINDOW_H__
#define __POPUPWINDOW_H__

#include <string>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
    class PopupWindow
    {
    private:
        std::string mName;
        float mX;
        float mY;
        float mWidth;
        float mHeight;
        bool mOpen;

    public:
        PopupWindow();
        PopupWindow(const std::string name, float x, float y, float width, float height);
        virtual ~PopupWindow() = 0;

        void draw(EditorClipboard& clipboard, bool isOpenedThisFrame);

        virtual void init(EditorClipboard& clipboard) = 0;
        virtual void update(EditorClipboard& clipboard) = 0;
    };
} // namespace PhysicsEditor

#endif