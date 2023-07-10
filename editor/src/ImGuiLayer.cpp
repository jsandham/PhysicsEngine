#include "../include/ImGuiLayer.h"

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include <stdio.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

#include "ImGuizmo.h"

#include "imgui_impl_dx11.h"
#include "imgui_impl_opengl3.h"

//#include "imgui_impl_win32.h"

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
    //// Cleanup
    //switch (PhysicsEngine::RenderContext::getRenderAPI())
    //{
    //case PhysicsEngine::RenderAPI::OpenGL:
    //    ImGui_ImplOpenGL3_Shutdown();
    //    break;
    //case PhysicsEngine::RenderAPI::DirectX:
    //    ImGui_ImplDX11_Shutdown();
    //    break;
    //}


 



    // Cleanup
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    
    ImGui::DestroyContext();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void ImGuiLayer::init()
{
    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD) != 0)
    {
        printf("Error: SDL_Init(): %s\n", SDL_GetError());
        return;
    }

    // Enable native IME.
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

    // Create window with SDL_Renderer graphics context
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN);
    window = SDL_CreateWindow("Dear ImGui SDL3+SDL_Renderer example", 1280, 720, window_flags);
    if (window == nullptr)
    {
        printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
        return;
    }
    renderer = SDL_CreateRenderer(window, NULL, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr)
    {
        SDL_Log("Error: SDL_CreateRenderer(): %s\n", SDL_GetError());
        return;
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);








    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    PhysicsEngine::Application& app = PhysicsEngine::Application::get();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    //switch (PhysicsEngine::RenderContext::getRenderAPI())
    //{
    //case PhysicsEngine::RenderAPI::OpenGL:
    //    // Init OpenGL Imgui Implementation
    //    // GL 3.0 + GLSL 130
    //    ImGui_ImplOpenGL3_Init("#version 330");
    //    break;
    //case PhysicsEngine::RenderAPI::DirectX:
    //    ImGui_ImplDX11_Init(PhysicsEngine::DirectXRenderContext::get()->getD3DDevice(), PhysicsEngine::DirectXRenderContext::get()->getD3DDeviceContext());
    //    break;
    //}

    // enable docking
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
}

void ImGuiLayer::begin()
{
    //// start the Dear ImGui frame
    //switch (PhysicsEngine::RenderContext::getRenderAPI())
    //{
    //case PhysicsEngine::RenderAPI::OpenGL:
    //    ImGui_ImplOpenGL3_NewFrame();
    //    break;
    //case PhysicsEngine::RenderAPI::DirectX:
    //    ImGui_ImplDX11_NewFrame();
    //    break;
    //}

    // Start the Dear ImGui frame
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
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
    // Rendering
    ImGui::Render();
    //SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    SDL_SetRenderDrawColor(renderer, (Uint8)(clear_color.x * 255), (Uint8)(clear_color.y * 255), (Uint8)(clear_color.z * 255), (Uint8)(clear_color.w * 255));
    SDL_RenderClear(renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData());
    SDL_RenderPresent(renderer);

    /*PhysicsEngine::Renderer::bindBackBuffer();
    switch (PhysicsEngine::RenderContext::getRenderAPI())
    {
    case PhysicsEngine::RenderAPI::OpenGL:
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        break;
    case PhysicsEngine::RenderAPI::DirectX:
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        break;
    }
    PhysicsEngine::Renderer::unbindBackBuffer();*/

    ImGui::EndFrame();
}

bool ImGuiLayer::quit()
{
    return false;
}








//#include "../include/ImGuiLayer.h"
//
//#include "imgui.h"
//#include "ImGuizmo.h"
//
//#include "imgui_impl_dx11.h"
//#include "imgui_impl_opengl3.h"
//
//#include "imgui_impl_win32.h"
//
//#include <core/Application.h>
//#include <graphics/Renderer.h>
//#include <graphics/RenderContext.h>
//#include <graphics/platform/directx/DirectXRenderContext.h>
//#include <windows.h>
//
//using namespace PhysicsEditor;
//
//ImGuiLayer::ImGuiLayer() : PhysicsEngine::Layer("Imgui")
//{
//
//}
//
//ImGuiLayer::~ImGuiLayer()
//{
//    // Cleanup
//    switch (PhysicsEngine::RenderContext::getRenderAPI())
//    {
//    case PhysicsEngine::RenderAPI::OpenGL:
//        ImGui_ImplOpenGL3_Shutdown();
//        break;
//    case PhysicsEngine::RenderAPI::DirectX:
//        ImGui_ImplDX11_Shutdown();
//        break;
//    }
//
//    ImGui_ImplWin32_Shutdown();
//    ImGui::DestroyContext();
//}
//
//void ImGuiLayer::init()
//{
//    // Setup Dear ImGui binding
//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//
//    PhysicsEngine::Application& app = PhysicsEngine::Application::get();
//
//    // Init Win32
//    ImGui_ImplWin32_Init(static_cast<HWND>(app.getWindow().getNativeWindow()));
//
//    switch (PhysicsEngine::RenderContext::getRenderAPI())
//    {
//    case PhysicsEngine::RenderAPI::OpenGL:
//        // Init OpenGL Imgui Implementation
//        // GL 3.0 + GLSL 130
//        ImGui_ImplOpenGL3_Init("#version 330");
//        break;
//    case PhysicsEngine::RenderAPI::DirectX:
//        ImGui_ImplDX11_Init(PhysicsEngine::DirectXRenderContext::get()->getD3DDevice(), PhysicsEngine::DirectXRenderContext::get()->getD3DDeviceContext());
//        break;
//    }
//
//    // enable docking
//    ImGuiIO& io = ImGui::GetIO();
//    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//}
//
//void ImGuiLayer::begin()
//{
//    // start the Dear ImGui frame
//    switch (PhysicsEngine::RenderContext::getRenderAPI())
//    {
//    case PhysicsEngine::RenderAPI::OpenGL:
//        ImGui_ImplOpenGL3_NewFrame();
//        break;
//    case PhysicsEngine::RenderAPI::DirectX:
//        ImGui_ImplDX11_NewFrame();
//        break;
//    }
//
//    ImGui_ImplWin32_NewFrame();
//    ImGui::NewFrame();
//    ImGuizmo::BeginFrame();
//
//    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
//        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
//        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
//        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
//    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
//
//    ImGuiViewport* viewport = ImGui::GetMainViewport();
//    ImGui::SetNextWindowPos(viewport->Pos);
//    ImGui::SetNextWindowSize(viewport->Size);
//    ImGui::SetNextWindowViewport(viewport->ID);
//
//    static bool p_open = true;
//    ImGui::Begin("Root Window", &p_open, window_flags);
//    ImGui::DockSpace(ImGui::GetID("Dockspace"), ImVec2(0.0f, 0.0f), dockspace_flags);
//}
//
//void ImGuiLayer::update(const PhysicsEngine::Time& time)
//{
//    ImGui::ShowDemoWindow();
//    //ImGui::ShowMetricsWindow();
//    //ImGui::ShowStyleEditor();
//}
//
//void ImGuiLayer::end()
//{
//    // end dockspace
//    ImGui::End();
//
//    // imgui render calls
//    ImGui::Render();
//
//    PhysicsEngine::Renderer::bindBackBuffer();
//    switch (PhysicsEngine::RenderContext::getRenderAPI())
//    {
//    case PhysicsEngine::RenderAPI::OpenGL:
//        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//        break;
//    case PhysicsEngine::RenderAPI::DirectX:
//        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
//        break;
//    }
//    PhysicsEngine::Renderer::unbindBackBuffer();
//
//    ImGui::EndFrame();
//}
//
//bool ImGuiLayer::quit()
//{
//    return false;
//}