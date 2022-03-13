#ifndef POPUPWINDOW_H__
#define POPUPWINDOW_H__

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

        void draw(Clipboard& clipboard, bool isOpenedThisFrame);

        void close();

        virtual void init(Clipboard& clipboard) = 0;
        virtual void update(Clipboard& clipboard) = 0;
    };
} // namespace PhysicsEditor

#endif