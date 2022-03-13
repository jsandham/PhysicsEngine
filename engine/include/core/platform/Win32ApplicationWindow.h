#ifndef WIN32_APPLICATION_WINDOW_H__
#define WIN32_APPLICATION_WINDOW_H__

#define NOMINMAX

#include <windows.h>

#include "../ApplicationWindow.h"

namespace PhysicsEngine
{
	class Win32ApplicationWindow : public ApplicationWindow
	{
	private:
		WNDCLASS mWC;
		HWND mWindow;

		std::string mTitle;
		bool mRunning;

	public:
		Win32ApplicationWindow(const std::string& title, int width, int height);
		~Win32ApplicationWindow();

		void update() override;
        void turnVsyncOn() override;
        void turnVsyncOff() override;

		int getWidth() const override;
		int getHeight() const override;

		bool isRunning() const override;
		bool isMinimized() const override;

		void* getNativeWindow() const override;

	private:
		void init(const std::string& title, int width, int height);
		void cleanup();
	};
}

#endif