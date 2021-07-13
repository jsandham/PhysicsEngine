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

void SpriteRendererDrawer::render(Clipboard& clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("SpriteRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SpriteRenderer* spriteRenderer = clipboard.getWorld()->getComponentById<SpriteRenderer>(id);

        ImGui::Text(("EntityId: " + spriteRenderer->getEntityId().toString()).c_str());
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
        color[0] = spriteRenderer->mColor.r;
        color[1] = spriteRenderer->mColor.g;
        color[2] = spriteRenderer->mColor.b;
        color[3] = spriteRenderer->mColor.a;

        if (ImGui::ColorEdit4("color", &color[0]))
        {
            spriteRenderer->mColor.r = color[0];
            spriteRenderer->mColor.g = color[1];
            spriteRenderer->mColor.b = color[2];
            spriteRenderer->mColor.a = color[3];
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

        ImGui::TreePop();
    }
}