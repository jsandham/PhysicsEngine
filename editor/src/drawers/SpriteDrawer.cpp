#include "../../include/drawers/SpriteDrawer.h"

#include "core/Sprite.h"
#include "core/Texture2D.h"

using namespace PhysicsEditor;

SpriteDrawer::SpriteDrawer()
{
    
}

SpriteDrawer::~SpriteDrawer()
{
  
}

void SpriteDrawer::render(Clipboard& clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Sprite* sprite = clipboard.getWorld()->getAssetById<Sprite>(id);

    Guid textureId = sprite->getTextureId();

    std::string textureName = "None (Texture)";
    if (textureId.isValid())
    {
        textureName = textureId.toString();
    }

    bool releaseTriggered = false;
    bool clearClicked = false;
    bool isClicked = ImGui::Slot("Texture", textureName, &releaseTriggered, &clearClicked);

    if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Texture2D)
    {
        textureId = clipboard.getDraggedId();
        clipboard.clearDraggedItem();

        sprite->setTextureId(textureId);
    }

    if (isClicked)
    {
        clipboard.setSelectedItem(InteractionType::Texture2D, textureId);
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}