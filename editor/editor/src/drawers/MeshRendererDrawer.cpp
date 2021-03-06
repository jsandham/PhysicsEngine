#include "../../include/drawers/MeshRendererDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/MeshRenderer.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{
}

MeshRendererDrawer::~MeshRendererDrawer()
{
}

void MeshRendererDrawer::render(EditorClipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        MeshRenderer *meshRenderer = clipboard.getWorld()->getComponentById<MeshRenderer>(id);

        ImGui::Text(("EntityId: " + meshRenderer->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        // Mesh
        Guid meshId = meshRenderer->getMesh();

        std::string meshName = "None (Mesh)";
        if (meshId != Guid::INVALID)
        {
            meshName = meshId.toString();
        }

        bool slotFilled = false;
        bool isClicked =
            ImGui::Slot("Mesh", meshName, clipboard.getDraggedType() == InteractionType::Mesh, &slotFilled);

        if (slotFilled)
        {
            meshId = clipboard.getDraggedId();
            clipboard.clearDraggedItem();

            meshRenderer->setMesh(meshId);
        }

        if (isClicked)
        {
            clipboard.setSelectedItem(InteractionType::Mesh, meshId);
        }

        bool isStatic = meshRenderer->mIsStatic;
        if (ImGui::Checkbox("Is Static?", &isStatic))
        {
            Undo::recordComponent(meshRenderer);

            meshRenderer->mIsStatic = isStatic;
        }

        bool enabled = meshRenderer->mEnabled;
        if (ImGui::Checkbox("Enabled?", &enabled))
        {
            Undo::recordComponent(meshRenderer);

            meshRenderer->mEnabled = enabled;
        }

        // Materials
        int materialCount = meshRenderer->mMaterialCount;
        const int increment = 1;
        ImGui::PushItemWidth(80);
        if (ImGui::InputScalar("Material Count", ImGuiDataType_S32, &materialCount, &increment, NULL, "%d"))
        {
            materialCount = std::max(0, std::min(materialCount, 8));

            Undo::recordComponent(meshRenderer);

            meshRenderer->mMaterialCount = materialCount;
        }
        ImGui::PopItemWidth();

        Guid materialIds[8];
        for (int i = 0; i < materialCount; i++)
        {
            materialIds[i] = meshRenderer->getMaterial(i);

            std::string materialName = "None (Material)";
            if (materialIds[i] != PhysicsEngine::Guid::INVALID)
            {
                materialName = materialIds[i].toString();
            }

            bool materialSlotFillable = clipboard.getDraggedType() == InteractionType::Material;
            bool materialSlotFilled = false;
            bool materialIsClicked = ImGui::Slot("Material", materialName, materialSlotFillable, &materialSlotFilled);

            if (materialSlotFilled)
            {
                materialIds[i] = clipboard.getDraggedId();
                clipboard.clearDraggedItem();

                meshRenderer->setMaterial(materialIds[i], i);
            }

            if (materialIsClicked)
            {
                Log::info((materialName + "\n").c_str());

                clipboard.setSelectedItem(InteractionType::Material, materialIds[i]);
            }

            // this current is always getting called when you click on an entity in the hierarchy causing the scene to
            // be dirtied
            // CommandManager::addCommand(new ChangePropertyCommand<Guid>(&meshRenderer->materialIds[i], materialIds[i],
            // &scene.isDirty));
        }

        ImGui::TreePop();
    }
}