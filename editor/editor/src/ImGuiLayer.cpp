#include "../include/ImGuiLayer.h"

#include "../include/imgui/imgui_styles.h"
#include "ImGuizmo.h"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"

#include "../include/IconsFontAwesome4.h"

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
}

void ImGuiLayer::begin()
{
    // start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

void ImGuiLayer::update()
{

}

void ImGuiLayer::end()
{
    // imgui render calls
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    ImGui::EndFrame();
}