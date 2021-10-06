#pragma once
#include "Application.h"

extern PhysicsEngine::Application* PhysicsEngine::createApplication();

#ifdef PHYSICSENGINE_PLATFORM_WINDOWS

#include <windows.h>

int main(int argc, char** argv)
{
	auto application = PhysicsEngine::createApplication();

	application->run();

	delete application;

	return 0;
}

#endif
