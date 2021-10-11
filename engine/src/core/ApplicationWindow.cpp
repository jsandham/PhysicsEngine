#include "../../include/core/ApplicationWindow.h"
#include "../../include/core/PlatformDetection.h"

#ifdef PHYSICSENGINE_PLATFORM_WIN32
#include "../../include/core/platform/Win32ApplicationWindow.h"
#endif

using namespace PhysicsEngine;

ApplicationWindow::ApplicationWindow()
{

}

ApplicationWindow::~ApplicationWindow()
{

}

ApplicationWindow* ApplicationWindow::createApplicationWindow(const std::string& title, int width, int height)
{
#ifdef PHYSICSENGINE_PLATFORM_WIN32
	return new Win32ApplicationWindow(title, width, height);
#endif

	return nullptr;
}