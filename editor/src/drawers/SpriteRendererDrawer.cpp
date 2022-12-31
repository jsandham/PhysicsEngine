#include "../../include/drawers/SpriteRendererDrawer.h"

#include "components/SpriteRenderer.h"

#include "imgui.h"
#include "imgui_internal.h"

//#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

SpriteRendererDrawer::SpriteRendererDrawer()
{
}

SpriteRendererDrawer::~SpriteRendererDrawer()
{
}

void SpriteRendererDrawer::render(Clipboard& clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("SpriteRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SpriteRenderer* spriteRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<SpriteRenderer>(id);

        if (spriteRenderer != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            // Sprite
            Sprite* sprite = clipboard.getWorld()->getAssetByGuid<Sprite>(spriteRenderer->getSprite());

            ImVec2 windowSize = ImGui::GetWindowSize();
            windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

            if (ImGui::ButtonEx((sprite == nullptr ? "None (Sprite)" : sprite->getName()).c_str(), ImVec2(windowSize.x, 0)))
            {
                clipboard.setSelectedItem(InteractionType::Sprite, sprite->getGuid());
            }

            if (ImGui::BeginDragDropTarget())
            {
                const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SPRITE");
                if (payload != nullptr)
                {
                    const PhysicsEngine::Guid* data = static_cast<const PhysicsEngine::Guid*>(payload->Data);

                    spriteRenderer->setSprite(*data);
                }
                ImGui::EndDragDropTarget();
            }

            ImVec2 size = ImGui::GetItemRectSize();
            ImVec2 position = ImGui::GetItemRectMin();

            ImVec2 topLeft = position;
            ImVec2 topRight = ImVec2(position.x + size.x, position.y);
            ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
            ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);

            ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
            ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
            ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
            ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);

            size.x += position.x;
            size.y += position.y;

            if (ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly))
            {
                ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
            }

            ImGui::SameLine();
            ImGui::Text("Sprite");

            float color[4];
            color[0] = spriteRenderer->mColor.mR;
            color[1] = spriteRenderer->mColor.mG;
            color[2] = spriteRenderer->mColor.mB;
            color[3] = spriteRenderer->mColor.mA;

            if (ImGui::ColorEdit4("color", &color[0]))
            {
                spriteRenderer->mColor.mR = color[0];
                spriteRenderer->mColor.mG = color[1];
                spriteRenderer->mColor.mB = color[2];
                spriteRenderer->mColor.mA = color[3];
            }

            bool flipX = spriteRenderer->mFlipX;
            if (ImGui::Checkbox("Flip X?", &flipX))
            {
                spriteRenderer->mFlipX = flipX;
            }

            bool flipY = spriteRenderer->mFlipY;
            if (ImGui::Checkbox("Flip Y?", &flipY))
            {
                spriteRenderer->mFlipY = flipY;
            }

            bool isStatic = spriteRenderer->mIsStatic;
            if (ImGui::Checkbox("Is Static?", &isStatic))
            {
                spriteRenderer->mIsStatic = isStatic;
            }

            bool enabled = spriteRenderer->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                spriteRenderer->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}