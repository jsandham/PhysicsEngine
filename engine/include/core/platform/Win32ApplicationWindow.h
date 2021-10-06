#ifndef WIN32_APPLICATION_WINDOW_H__
#define WIN32_APPLICATION_WINDOW_H__

#pragma warning(disable : 4996)
#define _CRTDBG_MAP_ALLOC

#include <windows.h>

#include "../ApplicationWindow.h"
#include "../ApplicationGraphicsContext.h"

namespace PhysicsEngine
{
	class Win32ApplicationWindow : public ApplicationWindow
	{
	private:
		WNDCLASS wc;
		HWND g_hwnd;
		HWND prevActiveWindow;
		HWND activeWindow;
		MSG msg;

		std::string mTitle;
		int mWidth;
		int mHeight;

		ApplicationGraphicsContext* mContext;

	public:
		Win32ApplicationWindow(const std::string& title, int width, int height);
		~Win32ApplicationWindow();

		void update() override;

		int getWidth() const override;
		int getHeight() const override;

	private:
		void init(const std::string& title, int width, int height);
		void cleanup();
	};
}

#endif