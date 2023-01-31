#include <filesystem>

#include "../include/EditorLayer.h"
#include "../include/ProjectDatabase.h"

#include "imgui.h"

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

void EditorLayer::update(const PhysicsEngine::Time& time)
{
    ProjectDatabase::update(mClipboard.getWorld());

    mMenuBar.update(mClipboard);

    mHierarchy.draw(mClipboard, mMenuBar.isOpenHierarchyCalled());
    mInspector.draw(mClipboard, mMenuBar.isOpenInspectorCalled());
    mConsole.draw(mClipboard, mMenuBar.isOpenConsoleCalled());
    mProjectView.draw(mClipboard, mMenuBar.isOpenProjectViewCalled());
    mSceneView.draw(mClipboard, mMenuBar.isOpenSceneViewCalled());

    mDebugOverlay.update(mClipboard);

    mClipboard.mTime = time;
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