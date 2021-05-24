#include "../include/Editor.h"
#include "../include/Undo.h"
#include "../include/imgui/imgui_styles.h"
#include "../include/IconsFontAwesome4.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "core/Guid.h"

using namespace PhysicsEditor;

Editor::Editor()
{
}

Editor::~Editor()
{
}

void Editor::init()
{
    // enable docking
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup style
    ImGui::StyleColorsCorporate();

    io.Fonts->AddFontDefault();

    ImFontConfig config;
    config.MergeMode = true;
    config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    io.Fonts->AddFontFromFileTTF("C:\\Users\\jsand\\Downloads\\fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
    io.Fonts->Build();

    mMenuBar.init(mClipboard);
    mInspector.init(mClipboard);
    mHierarchy.init(mClipboard);
    mSceneView.init(mClipboard);
    mProjectView.init(mClipboard);
    mConsole.init(mClipboard);
    mDebugOverlay.init(mClipboard);
}

void Editor::update()
{
    mClipboard.getLibrary().update(mClipboard.getWorld());

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    static bool p_open = true;
    ImGui::Begin("Root Window", &p_open, window_flags);
    ImGui::DockSpace(ImGui::GetID("Dockspace"), ImVec2(0.0f, 0.0f), dockspace_flags);

    //ImGui::ShowDemoWindow();
    // ImGui::ShowMetricsWindow();
    // ImGui::ShowStyleEditor();

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

    ImGui::End();

    if (mClipboard.getDraggedType() != InteractionType::None)
    {
        ImGui::GetForegroundDrawList()->AddText(ImGui::GetMousePos(), 0xFFFFFFFF, mClipboard.mDraggedPath.string().c_str());
    }

    Undo::updateUndoStack(mClipboard);
}