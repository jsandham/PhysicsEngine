#include "../../include/drawers/SceneDrawer.h"

using namespace PhysicsEditor;

SceneDrawer::SceneDrawer()
{

}

SceneDrawer::~SceneDrawer()
{

}

void SceneDrawer::render(Clipboard& clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Scene* scene = clipboard.getWorld()->getSceneByGuid(id);

        if (scene != nullptr)
        {
            ImGui::Text("Hello World");
        }
        else
        {
            ImGui::Text("Goodbye World");
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}