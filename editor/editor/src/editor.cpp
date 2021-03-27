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
}

void Editor::update()
{
    mClipboard.getLibrary().update();
    mClipboard.getLibrary().loadQueuedAssetsIntoWorld(mClipboard.getWorld());

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

    // ImGui::ShowDemoWindow();
    // ImGui::ShowMetricsWindow();
    // ImGui::ShowStyleEditor();

    mMenuBar.update(mClipboard);
    mHierarchy.draw(mClipboard, mMenuBar.isOpenHierarchyCalled());
    mInspector.draw(mClipboard, mMenuBar.isOpenInspectorCalled());
    mConsole.draw(mClipboard, mMenuBar.isOpenConsoleCalled());
    mProjectView.draw(mClipboard, mMenuBar.isOpenProjectViewCalled());
    mSceneView.draw(mClipboard, mMenuBar.isOpenSceneViewCalled());

    if (mClipboard.getDraggedType() != InteractionType::None)
    {
        ImVec2 size = ImVec2(5, 5);
        ImVec2 cursorPos = ImGui::GetMousePos();
        size.x += cursorPos.x;
        size.y += cursorPos.y;
        ImGui::GetForegroundDrawList()->AddRect(cursorPos, size, 0xFFFF0000);
    }

    ImGui::End();

    Undo::updateUndoStack(mClipboard);
}