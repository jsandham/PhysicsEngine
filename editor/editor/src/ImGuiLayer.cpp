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
    io.Fonts->AddFontFromFileTTF("fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
    io.Fonts->Build();
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
    // ImGui::ShowDemoWindow();
    // ImGui::ShowMetricsWindow();
    // ImGui::ShowStyleEditor();
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