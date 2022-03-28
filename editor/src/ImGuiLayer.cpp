#include "../include/ImGuiLayer.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"

#include <core/Application.h>
#include <windows.h>

using namespace PhysicsEditor;

ImGuiLayer::ImGuiLayer() : PhysicsEngine::Layer("Imgui")
{

}

ImGuiLayer::~ImGuiLayer()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    ImGui_ImplWin32_Shutdown();
}

void ImGuiLayer::init()
{
    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    PhysicsEngine::Application& app = PhysicsEngine::Application::get();

    // Init Win32
    ImGui_ImplWin32_Init(static_cast<HWND>(app.getWindow().getNativeWindow()));

    // Init OpenGL Imgui Implementation
    // GL 3.0 + GLSL 130
    ImGui_ImplOpenGL3_Init("#version 330");

    // enable docking
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
}

void ImGuiLayer::begin()
{
    // start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

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
}

void ImGuiLayer::update(const PhysicsEngine::Time& time)
{
    ImGui::ShowDemoWindow();
    //ImGui::ShowMetricsWindow();
    //ImGui::ShowStyleEditor();
}

void ImGuiLayer::end()
{
    // end dockspace
    ImGui::End();

    // imgui render calls
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    ImGui::EndFrame();
}

bool ImGuiLayer::quit()
{
    return false;
}