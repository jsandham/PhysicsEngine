#include "../include/EditorWin32.h"
#include "../include/imgui/imgui_styles.h"

#include "../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

EditorWin32::EditorWin32()
{
}

EditorWin32::~EditorWin32()
{
}

void EditorWin32::init(HWND window, int width, int height)
{
    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Init Win32
    ImGui_ImplWin32_Init(window);

    // Init OpenGL Imgui Implementation
    // GL 3.0 + GLSL 130
    ImGui_ImplOpenGL3_Init("#version 330");

    mEditor.init();
}

void EditorWin32::cleanUp()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    ImGui_ImplWin32_Shutdown();
}

void EditorWin32::update(HWND window, bool editorBecameActiveThisFrame)
{
    // start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    mEditor.update();

    // imgui render calls
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    ImGui::EndFrame();

    //commandManager.update();
}

bool EditorWin32::isQuitCalled() const
{
    return false;// editorMenu.isQuitClicked();
}