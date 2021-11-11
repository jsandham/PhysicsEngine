#include <core/Application.h>
#define PHYSICSENGINE_PLATFORM_WINDOWS 1
#include <core/EntryPoint.h>
#include "include/EditorLayer.h"
#include "include/ImGuiLayer.h"

namespace PhysicsEditor
{
	class EditorApplication : public PhysicsEngine::Application
	{
	private:
		ImGuiLayer mImguiLayer;
		EditorLayer mEditorLayer;

	public:
		EditorApplication() : Application("PhysicsEditor", 1536, 864)
		{
			pushLayer(&mImguiLayer);
			pushLayer(&mEditorLayer);
		}

		~EditorApplication()
		{
		}
	};
}

PhysicsEngine::Application* PhysicsEngine::createApplication()
{
	return new PhysicsEditor::EditorApplication();
}

#ifdef PHYSICSENGINE_PLATFORM_WINDOWS
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT Application_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
}
#endif
