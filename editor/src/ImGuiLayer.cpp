#include "../include/ImGuiLayer.h"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_sdlrenderer.h"
#include <stdio.h>

#if !SDL_VERSION_ATLEAST(2,0,17)
#error This backend requires SDL 2.0.17+ because of SDL_RenderGeometry() function
#endif

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

    // Cleanup
    ImGui_ImplSDLRenderer_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    
    ImGui::DestroyContext();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void ImGuiLayer::init()
{
    // Setup SDL
    // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
    // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
    /*if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)*/
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return;
    }

    // Setup window
    /*SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);*/
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    /*window = SDL_CreateWindow("Dear ImGui SDL2+SDL_Renderer example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);*/
    window = SDL_CreateWindow("Dear ImGui SDL2+SDL_Renderer example", 1280, 720, window_flags);

    // Setup SDL_Renderer instance
    renderer = SDL_CreateRenderer(window, "Test", SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
    if (renderer == NULL)
    {
        SDL_Log("Error creating SDL_Renderer!");
        return;
    }








    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    PhysicsEngine::Application& app = PhysicsEngine::Application::get();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer_Init(renderer);

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

    ImGui_ImplSDLRenderer_NewFrame();
    ImGui_ImplSDL2_NewFrame();
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
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    SDL_SetRenderDrawColor(renderer, (Uint8)(clear_color.x * 255), (Uint8)(clear_color.y * 255), (Uint8)(clear_color.z * 255), (Uint8)(clear_color.w * 255));
    SDL_RenderClear(renderer);
    ImGui_ImplSDLRenderer_RenderDrawData(ImGui::GetDrawData());
    SDL_RenderPresent(renderer);

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