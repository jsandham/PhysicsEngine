#ifndef APPLICATION_H__
#define APPLICATION_H__

#include <string>
#include <vector>
#include <chrono>

#include "ApplicationWindow.h"
#include "Layer.h"
#include "Time.h"

namespace PhysicsEngine
{
	class Application
	{
	private:
		ApplicationWindow* mWindow;

		std::vector<Layer*> mLayers;
		std::string mName;
		Time mTime;
		bool mRunning;
		bool mMinimized;

	public:
		Application(const std::string& name = "App", int width = 1920, int height = 1080);
		Application(const Application& other) = delete;
		Application& operator=(const Application& other) = delete;
		virtual ~Application();

		void run();
		void close();

		void pushLayer(Layer* layer);

		ApplicationWindow& getWindow();
		static Application& get() { return *mInstance; }

	private:
		static Application* mInstance;
	};

	// Defined by client
	Application* createApplication();
}

#endif