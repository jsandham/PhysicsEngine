#include "../../include/drawers/SpriteRendererDrawer.h"

#include "components/SpriteRenderer.h"

#include "imgui.h"

#include "../../include/imgui/imgui_extensions.h"

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
        SpriteRenderer* spriteRenderer = clipboard.getWorld()->getActiveScene()->getComponentById<SpriteRenderer>(id);

        if (spriteRenderer != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            // Sprite
            Guid spriteId = spriteRenderer->getSprite();

            std::string spriteName = "None (Sprite)";
            if (spriteId.isValid())
            {
                spriteName = spriteId.toString();
            }

            bool releaseTriggered = false;
            bool clearClicked = false;
            bool isClicked = ImGui::Slot("Sprite", spriteName, &releaseTriggered, &clearClicked);

            if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Sprite)
            {
                spriteId = clipboard.getDraggedId();
                clipboard.clearDraggedItem();

                spriteRenderer->setSprite(spriteId);
            }

            if (isClicked)
            {
                clipboard.setSelectedItem(InteractionType::Sprite, spriteId);
            }

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