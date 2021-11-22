#include <core/Application.h>
#define PHYSICSENGINE_PLATFORM_WINDOWS 1
#include <core/EntryPoint.h>
#include <core/GameLayer.h>

namespace Game
{
	class GameApplication : public PhysicsEngine::Application
	{
	private:
		PhysicsEngine::GameLayer mGame;

	public:
		GameApplication() : PhysicsEngine::Application("Game")
		{
			pushLayer(&mGame);
		}

		~GameApplication()
		{
		}
	};
}

PhysicsEngine::Application* PhysicsEngine::createApplication()
{
	return new Game::GameApplication();
}

#ifdef PHYSICSENGINE_PLATFORM_WINDOWS
#include <Windows.h>

LRESULT Application_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return 0;
}
#endif
