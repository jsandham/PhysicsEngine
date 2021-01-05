#include "../../include/views/Window.h"

using namespace PhysicsEditor;

Window::Window()
{

}

Window::~Window()
{

}

void Window::init(EditorClipboard& clipboard)
{
    windowActive = true;
}

void Window::update(EditorClipboard& clipboard, bool isOpenedThisFrame)
{
    if (isOpenedThisFrame)
    {
        windowActive = true;
    }
}