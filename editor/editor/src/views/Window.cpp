#include "../../include/views/Window.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

Window::Window() : mName("Default"), mOpen(true)
{
}

Window::Window(const std::string name) : mName(name), mOpen(true)
{
}

Window::~Window()
{
}

void Window::draw(EditorClipboard& clipboard, bool isOpenedThisFrame)
{
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

        update(clipboard);
    }

    ImGui::End();
}