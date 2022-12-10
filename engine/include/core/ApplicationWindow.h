#ifndef APPLICATION_WINDOW_H__
#define APPLICATION_WINDOW_H__

#include <string>

namespace PhysicsEngine
{
	class ApplicationWindow
	{
	public:
		ApplicationWindow();
		virtual ~ApplicationWindow() = 0;

		virtual void update() = 0;
        virtual void turnVsyncOn() = 0;
        virtual void turnVsyncOff() = 0;

		virtual int getWidth() const = 0;
		virtual int getHeight() const = 0;

		virtual bool isRunning() const = 0;
		virtual bool isMinimized() const = 0;

		virtual void* getNativeWindow() const = 0;

		static ApplicationWindow* createApplicationWindow(const std::string& title, int width, int height);
	};
}

#endif