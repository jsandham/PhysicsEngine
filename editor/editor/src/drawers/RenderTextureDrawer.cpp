#include "../../include/drawers/RenderTextureDrawer.h"
#include "../../include/EditorClipboard.h"
#include "../../include/EditorCommands.h"

#include "core/RenderTexture.h"

#include "imgui.h"

using namespace PhysicsEditor;

RenderTextureDrawer::RenderTextureDrawer()
{
}

RenderTextureDrawer::~RenderTextureDrawer()
{
}

void RenderTextureDrawer::render(Clipboard& clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    RenderTexture* texture = clipboard.getWorld()->getAssetById<RenderTexture>(id);

    ImGui::Separator();

    // Draw texture child window
    {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar;
        ImGui::BeginChild("DrawTextureWindow",
            ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true,
            window_flags);

        ImGui::Image((void*)(intptr_t)texture->getNativeGraphicsColorTex(),
            ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::EndChild();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}