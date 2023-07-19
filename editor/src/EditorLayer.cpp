#include <filesystem>

#include "../include/EditorLayer.h"
#include "../include/ProjectDatabase.h"

using namespace PhysicsEditor;

EditorLayer::EditorLayer() : Layer("Editor")
{
}

EditorLayer::~EditorLayer()
{
}

void EditorLayer::init()
{
	std::filesystem::path cwd = std::filesystem::current_path();
	std::filesystem::path dataPath = cwd / "data";

	mClipboard.getWorld()->loadAssetsInPath(dataPath);

	mMenuBar.init(mClipboard);
	mInspector.init(mClipboard);
	mHierarchy.init(mClipboard);
	mSceneView.init(mClipboard);
	mProjectView.init(mClipboard);
	mConsole.init(mClipboard);
	mDebugOverlay.init(mClipboard);
}

void EditorLayer::update()
{
	ProjectDatabase::update(mClipboard.getWorld());

	mMenuBar.update(mClipboard);

	mHierarchy.update(mClipboard, mMenuBar.isOpenHierarchyCalled());
	mInspector.update(mClipboard, mMenuBar.isOpenInspectorCalled());
	mConsole.update(mClipboard, mMenuBar.isOpenConsoleCalled());
	mProjectView.update(mClipboard, mMenuBar.isOpenProjectViewCalled());
	mSceneView.update(mClipboard, mMenuBar.isOpenSceneViewCalled());

	mDebugOverlay.update(mClipboard);

	mClipboard.mTime = PhysicsEngine::getTime();//time;
}

void EditorLayer::begin()
{

}

void EditorLayer::end()
{

}

bool EditorLayer::quit()
{
	return mMenuBar.isQuitClicked();
}