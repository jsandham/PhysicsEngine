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

void SpriteDrawer::render(Clipboard& clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    //Sprite* sprite = clipboard.getWorld()->getAssetByGuid<Sprite>(id);


    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}