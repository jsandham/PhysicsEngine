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
	mMinimized = false;

	mTime.mStartTime = 0;
	mTime.mEndTime = 0;
	mTime.mDeltaTime = 0;
	mTime.mFrameCount = 0;

	mWindow = ApplicationWindow::createApplicationWindow(name, 1920, 1080);
}

Application::~Application()
{
	delete mWindow;
}

void Application::run()
{
	auto app_start = std::chrono::high_resolution_clock::now();

	while (mRunning)
	{
		auto start = std::chrono::high_resolution_clock::now();

		if (!mMinimized)
		{
			for (Layer* layer : mLayers)
			{
				layer->begin();
			}

			for (Layer* layer : mLayers)
			{
				layer->update(mTime);
			}

			for (Layer* layer : mLayers)
			{
				layer->end();
			}
		}

		mWindow->update();

		mRunning = mWindow->isRunning();
		mMinimized = mWindow->isMinimized();

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> start_time = start - app_start;
		std::chrono::duration<double> end_time = end - app_start;
		std::chrono::duration<double> elapsed_time = end - start;

		mTime.mStartTime = start_time.count();
		mTime.mEndTime = end_time.count();
		mTime.mDeltaTime = elapsed_time.count();
		mTime.mFrameCount++;
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