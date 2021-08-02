#include "../../include/views/PopupWindow.h"

#include "imgui.h"

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

void PopupWindow::draw(Clipboard& clipboard, bool isOpenedThisFrame)
{
    if (isOpenedThisFrame)
    {
        ImGui::SetNextWindowPos(ImVec2(mX, mY));
        ImGui::SetNextWindowSize(ImVec2(mWidth, mHeight));

        ImGui::OpenPopup(mName.c_str());
        mOpen = true;
    }

    if (ImGui::BeginPopupModal(mName.c_str(), &mOpen, ImGuiWindowFlags_NoResize))
    {
        update(clipboard);

        ImGui::EndPopup();
    }
}