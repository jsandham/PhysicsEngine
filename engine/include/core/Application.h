#ifndef APPLICATION_H__
#define APPLICATION_H__

#include <string>

#include "ApplicationWindow.h"
#include "ApplicationGraphicsContext.h"
#include "Layer.h"

namespace PhysicsEngine
{
	class Application
	{
	private:
		ApplicationWindow* mWindow;

		Layer* mLayer;
		std::string mName;
		bool mRunning;

	public:
		Application(const std::string& name = "App");
		virtual ~Application();

		void run();
		void close();

		void pushLayer(Layer* layer);

	private:
		static Application* mInstance;
	};

	// Defined by client
	Application* createApplication();
}

#endif