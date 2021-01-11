#include "../../include/views/PopupWindow.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

PopupWindow::PopupWindow() : mName("Default"), mX(0), mY(0), mWidth(200), mHeight(200), mOpen(true)
{
}

PopupWindow::PopupWindow(const std::string name, float x, float y, float width, float height) : mName(name), mX(x), mY(y), mWidth(width), mHeight(height), mOpen(true)
{
}

PopupWindow::~PopupWindow()
{
}

void PopupWindow::draw(EditorClipboard& clipboard, bool isOpenedThisFrame)
{
    if (isOpenedThisFrame)
    {
        ImGui::SetNextWindowSizeConstraints(ImVec2(mX, mY), ImVec2(mWidth, mHeight));
        ImGui::OpenPopup(mName.c_str());
    }

    if (mOpen && ImGui::BeginPopupModal(mName.c_str(), &mOpen))
    {
        update(clipboard);

        ImGui::EndPopup();
    }
}