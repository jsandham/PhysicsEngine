#include "../../include/views/Window.h"

#include "imgui.h"

using namespace PhysicsEditor;

Window::Window() : mName("Default"), mOpen(true), mFocused(false), mHovered(false)
{
    mWindowPos = ImVec2(0, 0);
    mContentMin = ImVec2(0, 0);
    mContentMax = ImVec2(0, 0);
}

Window::Window(const std::string name) : mName(name), mOpen(true), mFocused(false), mHovered(false)
{
    mWindowPos = ImVec2(0, 0);
    mContentMin = ImVec2(0, 0);
    mContentMax = ImVec2(0, 0);
}

Window::~Window()
{
}

void Window::draw(Clipboard& clipboard, bool isOpenedThisFrame)
{
    mWindowPos.x = 0;
    mWindowPos.y = 0;
    mContentMax.x = 0;
    mContentMax.y = 0;
    mContentMin.x = 0;
    mContentMin.y = 0;
    mFocused = false;
    mHovered = false;

    if (isOpenedThisFrame)
    {
        mOpen = true;
    }

    if (!mOpen)
    {
        return;
    }

    if (ImGui::Begin(mName.c_str(), &mOpen))
    {
        if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
        {
            ImGui::SetWindowFocus(mName.c_str());
        }
    }

    mWindowPos = ImGui::GetWindowPos();
    mContentMin = ImGui::GetWindowContentRegionMin();
    mContentMax = ImGui::GetWindowContentRegionMax();

    mContentMin.x += getWindowPos().x;
    mContentMin.y += getWindowPos().y;
    mContentMax.x += getWindowPos().x;
    mContentMax.y += getWindowPos().y;

    mFocused = ImGui::IsWindowFocused();
    mHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    if (mOpen)
    {
        update(clipboard);
    }

    ImGui::End();
}

ImVec2 Window::getWindowPos() const
{
    return mWindowPos;
}

ImVec2 Window::getContentMin() const
{
    return mContentMin;
}

ImVec2 Window::getContentMax() const
{
    return mContentMax;
}

bool Window::isOpen() const
{
    return mOpen;
}

bool Window::isFocused() const
{
    return mFocused;
}

bool Window::isHovered() const
{
    return mHovered;
}