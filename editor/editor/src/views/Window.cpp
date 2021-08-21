#include "../../include/views/Window.h"

#include "imgui.h"

using namespace PhysicsEditor;

Window::Window() : mName("Default"), mOpen(true), mFocused(false), mHovered(false)
{
    mWindowPos = ImVec2(0, 0);
    mWindowWidth = 0.0f;
    mWindowHeight = 0.0f;
    mContentMin = ImVec2(0, 0);
    mContentMax = ImVec2(0, 0);

    mOpenedLastFrame = false;
    mFocusedLastFrame = false;
    mHoveredLastFrame = false;
}

Window::Window(const std::string name) : mName(name), mOpen(true), mFocused(false), mHovered(false)
{
    mWindowPos = ImVec2(0, 0);
    mWindowWidth = 0.0f;
    mWindowHeight = 0.0f;
    mContentMin = ImVec2(0, 0);
    mContentMax = ImVec2(0, 0);

    mOpenedLastFrame = false;
    mFocusedLastFrame = false;
    mHoveredLastFrame = false;
}

Window::~Window()
{
}

void Window::draw(Clipboard& clipboard, bool isOpenedThisFrame, float alpha, ImGuiWindowFlags flags)
{
    mWindowPos = ImVec2(0, 0);
    mWindowWidth = 0.0f;
    mWindowHeight = 0.0f;
    mContentMax = ImVec2(0, 0);
    mContentMin = ImVec2(0, 0);

    mOpenedLastFrame = mOpen;
    mFocusedLastFrame = mFocused;
    mHoveredLastFrame = mHovered;
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

    ImGui::SetNextWindowBgAlpha(alpha);

    if (ImGui::Begin(mName.c_str(), &mOpen, flags))
    {
        if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
        {
            ImGui::SetWindowFocus(mName.c_str());
        }
    }

    mWindowPos = ImGui::GetWindowPos();
    mWindowWidth = ImGui::GetWindowWidth();
    mWindowHeight = ImGui::GetWindowHeight();
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

    if (ImGui::IsWindowHovered()) {
        ImGui::GetForegroundDrawList()->AddRect(mContentMin, mContentMax, 0xFFFF0000);
    }

    ImGui::End();
}

void Window::close()
{
    mOpen = false;
}

ImVec2 Window::getWindowPos() const
{
    return mWindowPos;
}

float Window::getWindowWidth() const
{
    return mWindowWidth;
}

float Window::getWindowHeight() const
{
    return mWindowHeight;
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

bool Window::openedThisFrame() const
{
    return !mOpenedLastFrame && mOpen;
}

bool Window::closedThisFrame() const
{
    return mOpenedLastFrame && !mOpen;
}

bool Window::focusedThisFrame() const
{
    return !mFocusedLastFrame && mFocused;
}

bool Window::hoveredThisFrame() const
{
    return !mHoveredLastFrame && mHovered;
}

bool Window::unfocusedThisFrame() const
{
    return mFocusedLastFrame && !mFocused;
}

bool Window::unhoveredThisFrame() const
{
    return mHoveredLastFrame && !mHovered;
}