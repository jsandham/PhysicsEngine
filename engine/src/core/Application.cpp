#include "../../include/core/Application.h"

#include<assert.h>

using namespace PhysicsEngine;

Application* Application::mInstance = nullptr;

Application::Application(const std::string& name)
{
	assert(mInstance == nullptr);
	mInstance = this;

	mName = name;
	mRunning = true;

	mWindow = ApplicationWindow::createApplicationWindow(name, 1920, 1080);
}

Application::~Application()
{
	delete mWindow;
}

void Application::run()
{
	while (mRunning)
	{
		for (Layer* layer : mLayers)
		{
			layer->begin();
		}

		for (Layer* layer : mLayers)
		{
			layer->update();
		}

		for (Layer* layer : mLayers)
		{
			layer->end();
		}

		mWindow->update();
	}
}

void Application::close()
{
	mRunning = false;
}

void Application::pushLayer(Layer* layer)
{
	mLayers.push_back(layer);

	layer->init();
}

ApplicationWindow& Application::getWindow()
{ 
	return *mWindow; 
}