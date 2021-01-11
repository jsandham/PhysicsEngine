//#include "../include/EditorToolbar.h"
//
//#include "imgui.h"
//#include "imgui_impl_opengl3.h"
//#include "imgui_impl_win32.h"
//#include "imgui_internal.h"
//
//#include "core/Guid.h"
//
//using namespace PhysicsEditor;
//
//EditorToolbar::EditorToolbar()  //rename to DockspaceWindow? MainWindow? RootWindow? Editor? DockingWindow?
//{
//}
//
//EditorToolbar::~EditorToolbar()
//{
//}
//
//void EditorToolbar::render(EditorClipboard &clipboard)
//{
//    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
//                                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
//                                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
//                                    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
//    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
//    
//    ImGuiViewport *viewport = ImGui::GetMainViewport();
//    ImGui::SetNextWindowPos(viewport->Pos);
//    ImGui::SetNextWindowSize(viewport->Size);
//    ImGui::SetNextWindowViewport(viewport->ID);
//
//    static bool p_open = true;
//    ImGui::Begin("Root Window", &p_open, window_flags);
//    ImGui::DockSpace(ImGui::GetID("Dockspace"), ImVec2(0.0f, 0.0f), dockspace_flags);
//
//    if (clipboard.getDraggedType() != InteractionType::None)
//    {
//        ImVec2 size = ImVec2(5, 5);
//        ImVec2 cursorPos = ImGui::GetMousePos();
//        size.x += cursorPos.x;
//        size.y += cursorPos.y;
//        ImGui::GetForegroundDrawList()->AddRect(cursorPos, size, 0xFFFF0000);
//    }
//
//    ImGui::End();
//}