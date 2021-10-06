#include "../../include/core/ApplicationWindow.h"
#include "../../include/core/platform/Win32ApplicationWindow.h"

using namespace PhysicsEngine;

ApplicationWindow::ApplicationWindow()
{

}

ApplicationWindow::~ApplicationWindow()
{

}

ApplicationWindow* ApplicationWindow::createApplicationWindow(const std::string& title, int width, int height)
{
//#ifdef PHYSICSENGINE_PLATFORM_WINDOWS
	return new Win32ApplicationWindow(title, width, height);
//#endif

	//return nullptr;
}