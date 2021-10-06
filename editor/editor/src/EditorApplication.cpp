#include <core/Application.h>
#define PHYSICSENGINE_PLATFORM_WINDOWS 1
#include <core/EntryPoint.h>
#include "../include/Editor.h"

namespace PhysicsEditor
{
	class EditorApplication : public PhysicsEngine::Application
	{
	private:
		Editor* mEditor;

	public:
		EditorApplication() : Application("PhysicsEditor")
		{
			mEditor = new Editor();
			pushLayer(mEditor);
		}

		~EditorApplication()
		{
			delete mEditor;
		}
	};
}

PhysicsEngine::Application* PhysicsEngine::createApplication()
{
	return new PhysicsEditor::EditorApplication();
}
