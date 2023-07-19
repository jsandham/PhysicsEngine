#pragma once
#include "Application.h"
#include "PlatformDetection.h"

extern PhysicsEngine::Application *PhysicsEngine::createApplication();

int main(int argc, char **argv)
{
    PhysicsEngine::Application *application = PhysicsEngine::createApplication();

    application->run();

    delete application;

    return 0;
}
