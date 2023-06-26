#include "../include/ImGuiLayer.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include "imgui_impl_dx11.h"
#include "imgui_impl_opengl3.h"

#include "imgui_impl_win32.h"

#include <core/Application.h>
#include <graphics/Renderer.h>
#include <graphics/RenderContext.h>
#include <graphics/platform/directx/DirectXRenderContext.h>
#include <windows.h>

using namespace PhysicsEditor;

ImGuiLayer::ImGuiLayer() : PhysicsEngine::Layer("Imgui")
{

}

ImGuiLayer::~ImGuiLayer()
{
    // Cleanup
    switch (PhysicsEngine::RenderContext::getRenderAPI())
    {
    case PhysicsEngine::RenderAPI::OpenGL:
        ImGui_ImplOpenGL3_Shutdown();
        break;
    case PhysicsEngine::RenderAPI::DirectX:
        ImGui_ImplDX11_Shutdown();
        break;
    }

    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::init()
{
    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    PhysicsEngine::Application& app = PhysicsEngine::Application::get();

    // Init Win32
    ImGui_ImplWin32_Init(static_cast<HWND>(app.getWindow().getNativeWindow()));

    switch (PhysicsEngine::RenderContext::getRenderAPI())
    {
    case PhysicsEngine::RenderAPI::OpenGL:
        // Init OpenGL Imgui Implementation
        // GL 3.0 + GLSL 130
        ImGui_ImplOpenGL3_Init("#version 330");
        break;
    case PhysicsEngine::RenderAPI::DirectX:
        ImGui_ImplDX11_Init(PhysicsEngine::DirectXRenderContext::get()->getD3DDevice(), PhysicsEngine::DirectXRenderContext::get()->getD3DDeviceContext());
        break;
    }

    // enable docking
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
}

void ImGuiLayer::begin()
{
    // start the Dear ImGui frame
    switch (PhysicsEngine::RenderContext::getRenderAPI())
    {
    case PhysicsEngine::RenderAPI::OpenGL:
        ImGui_ImplOpenGL3_NewFrame();
        break;
    case PhysicsEngine::RenderAPI::DirectX:
        ImGui_ImplDX11_NewFrame();
        break;
    }

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

    PhysicsEngine::Renderer::bindBackBuffer();
    switch (PhysicsEngine::RenderContext::getRenderAPI())
    {
    case PhysicsEngine::RenderAPI::OpenGL:
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        break;
    case PhysicsEngine::RenderAPI::DirectX:
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        break;
    }
    PhysicsEngine::Renderer::unbindBackBuffer();

    ImGui::EndFrame();
}

bool ImGuiLayer::quit()
{
    return false;
}