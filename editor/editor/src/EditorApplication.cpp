#include <core/Application.h>
#define PHYSICSENGINE_PLATFORM_WINDOWS 1
#include <core/EntryPoint.h>
#include "../include/Editor.h"
#include "../include/ImGuiLayer.h"

namespace PhysicsEditor
{
	class EditorApplication : public PhysicsEngine::Application
	{
	private:
		ImGuiLayer mImguiLayer;
		Editor mEditor;

	public:
		EditorApplication() : Application("PhysicsEditor")
		{
			pushLayer(&mImguiLayer);
			pushLayer(&mEditor);
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
LRESULT PhysicsEngine_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
}
#endif
