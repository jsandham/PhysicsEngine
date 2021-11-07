#pragma once
#include "PlatformDetection.h"
#include "Application.h"

extern PhysicsEngine::Application* PhysicsEngine::createApplication();

int main(int argc, char** argv)
{
	PhysicsEngine::Application* application = PhysicsEngine::createApplication();

	application->run();

	delete application;

	return 0;
}