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
