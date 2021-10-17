#include <filesystem>
#include <stack>

#include "../include/Editor.h"

#include "imgui.h"

#include "core/Guid.h"

using namespace PhysicsEditor;
namespace fs = std::filesystem;

Editor::Editor() : Layer("Editor")
{
}

Editor::~Editor()
{
}

void Editor::init()
{
    fs::path cwd = fs::current_path();
    fs::path dataPath = cwd / "data";

    if (fs::is_directory(dataPath))
    {
        std::stack<fs::path> stack;
        stack.push(dataPath);

        while (!stack.empty())
        {
            fs::path currentPath = stack.top();
            stack.pop();

            std::error_code error_code;
            for (const fs::directory_entry& entry : fs::directory_iterator(currentPath, error_code))
            {
                if (fs::is_directory(entry, error_code))
                {
                    stack.push(entry.path());
                }
                else if (fs::is_regular_file(entry, error_code))
                {
                    std::string extension = entry.path().extension().string();
                    if (extension == ".mesh" || 
                        extension == ".shader" ||
                        extension == ".material" ||
                        extension == ".texture")
                    {
                        fs::path relativeDataPath = entry.path().lexically_relative(fs::current_path());
                        mClipboard.getWorld()->loadAssetFromYAML(relativeDataPath.string());
                    }
                }
            }
        }
    }

    mMenuBar.init(mClipboard);
    mInspector.init(mClipboard);
    mHierarchy.init(mClipboard);
    mSceneView.init(mClipboard);
    mProjectView.init(mClipboard);
    mConsole.init(mClipboard);
    mDebugOverlay.init(mClipboard);
}

void Editor::update(const PhysicsEngine::Time& time)
{
    mClipboard.getLibrary().update(mClipboard.getWorld());

    mMenuBar.update(mClipboard);

    mHierarchy.draw(mClipboard, mMenuBar.isOpenHierarchyCalled());
    mInspector.draw(mClipboard, mMenuBar.isOpenInspectorCalled());
    mConsole.draw(mClipboard, mMenuBar.isOpenConsoleCalled());
    mProjectView.draw(mClipboard, mMenuBar.isOpenProjectViewCalled());
    mSceneView.draw(mClipboard, mMenuBar.isOpenSceneViewCalled());

    mClipboard.mHierarchyOpen = mHierarchy.isOpen();
    mClipboard.mHierarchyHovered = mHierarchy.isHovered();
    mClipboard.mHierarchyFocused = mHierarchy.isFocused();
    mClipboard.mHierarchyOpenedThisFrame = mHierarchy.openedThisFrame();
    mClipboard.mHierarchyHoveredThisFrame = mHierarchy.hoveredThisFrame();
    mClipboard.mHierarchyFocusedThisFrame = mHierarchy.focusedThisFrame();
    mClipboard.mHierarchyClosedThisFrame = mHierarchy.closedThisFrame();
    mClipboard.mHierarchyUnfocusedThisFrame = mHierarchy.unfocusedThisFrame();
    mClipboard.mHierarchyUnhoveredThisFrame = mHierarchy.unhoveredThisFrame();

    mClipboard.mInspectorOpen = mInspector.isOpen();
    mClipboard.mInspectorHovered = mInspector.isHovered();
    mClipboard.mInspectorFocused = mInspector.isFocused();
    mClipboard.mInspectorOpenedThisFrame = mInspector.openedThisFrame();
    mClipboard.mInspectorHoveredThisFrame = mInspector.hoveredThisFrame();
    mClipboard.mInspectorFocusedThisFrame = mInspector.focusedThisFrame();
    mClipboard.mInspectorClosedThisFrame = mInspector.closedThisFrame();
    mClipboard.mInspectorUnfocusedThisFrame = mInspector.unfocusedThisFrame();
    mClipboard.mInspectorUnhoveredThisFrame = mInspector.unhoveredThisFrame();

    mClipboard.mConsoleOpen = mConsole.isOpen();
    mClipboard.mConsoleHovered = mConsole.isHovered();
    mClipboard.mConsoleFocused = mConsole.isFocused();
    mClipboard.mConsoleOpenedThisFrame = mConsole.openedThisFrame();
    mClipboard.mConsoleHoveredThisFrame = mConsole.hoveredThisFrame();
    mClipboard.mConsoleFocusedThisFrame = mConsole.focusedThisFrame();
    mClipboard.mConsoleClosedThisFrame = mConsole.closedThisFrame();
    mClipboard.mConsoleUnfocusedThisFrame = mConsole.unfocusedThisFrame();
    mClipboard.mConsoleUnhoveredThisFrame = mConsole.unhoveredThisFrame();

    mClipboard.mProjectViewOpen = mProjectView.isOpen();
    mClipboard.mProjectViewHovered = mProjectView.isHovered();
    mClipboard.mProjectViewFocused = mProjectView.isFocused();
    mClipboard.mProjectViewOpenedThisFrame = mProjectView.openedThisFrame();
    mClipboard.mProjectViewHoveredThisFrame = mProjectView.hoveredThisFrame();
    mClipboard.mProjectViewFocusedThisFrame = mProjectView.focusedThisFrame();
    mClipboard.mProjectViewClosedThisFrame = mProjectView.closedThisFrame();
    mClipboard.mProjectViewUnfocusedThisFrame = mProjectView.unfocusedThisFrame();
    mClipboard.mProjectViewUnhoveredThisFrame = mProjectView.unhoveredThisFrame();

    mClipboard.mSceneViewOpen = mSceneView.isOpen();
    mClipboard.mSceneViewHovered = mSceneView.isHovered();
    mClipboard.mSceneViewFocused = mSceneView.isFocused();
    mClipboard.mSceneViewOpenedThisFrame = mSceneView.openedThisFrame();
    mClipboard.mSceneViewHoveredThisFrame = mSceneView.hoveredThisFrame();
    mClipboard.mSceneViewFocusedThisFrame = mSceneView.focusedThisFrame();
    mClipboard.mSceneViewClosedThisFrame = mSceneView.closedThisFrame();
    mClipboard.mSceneViewUnfocusedThisFrame = mSceneView.unfocusedThisFrame();
    mClipboard.mSceneViewUnhoveredThisFrame = mSceneView.unhoveredThisFrame();

    static ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoResize;

    static bool show_overlay = true;
    if (ImGui::IsKeyPressed(17, false))
    {
        show_overlay = !show_overlay;
    }

    if (show_overlay)
    {
        mDebugOverlay.draw(mClipboard, show_overlay, 0.35f, overlay_flags);
    }

    if (mClipboard.getDraggedType() != InteractionType::None)
    {
        ImGui::GetForegroundDrawList()->AddText(ImGui::GetMousePos(), 0xFFFFFFFF, mClipboard.mDraggedPath.string().c_str());
    }

    mClipboard.mTime = time;
}

void Editor::begin()
{

}

void Editor::end()
{

}