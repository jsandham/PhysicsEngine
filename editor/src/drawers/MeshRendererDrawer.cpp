#include "../../include/drawers/MeshRendererDrawer.h"

#include "components/MeshRenderer.h"

#include "imgui.h"

#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{
}

MeshRendererDrawer::~MeshRendererDrawer()
{
}

void MeshRendererDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        MeshRenderer *meshRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<MeshRenderer>(id);

        if (meshRenderer != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            // Mesh
            {
                Guid meshId = meshRenderer->getMesh();

                Mesh* mesh = clipboard.getWorld()->getAssetByGuid<Mesh>(meshId);

                ImGui::SlotData data;
                if (ImGui::Slot2("Mesh", meshId.isValid() ? mesh->getName()/*meshId.toString()*/ : "None (Mesh)", &data))
                {
                    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Mesh)
                    {
                        meshId = clipboard.getDraggedId();
                        clipboard.clearDraggedItem();

                        meshRenderer->setMesh(meshId);
                    }

                    if (data.isClicked && meshId.isValid())
                    {
                        clipboard.setSelectedItem(InteractionType::Mesh, meshId);
                    }

                    if (data.clearClicked)
                    {
                        meshRenderer->setMesh(Guid::INVALID);
                    }
                }
            }

            // Materials
            int materialCount = meshRenderer->mMaterialCount;
            const int increment = 1;
            ImGui::PushItemWidth(80);
            if (ImGui::InputScalar("Material Count", ImGuiDataType_S32, &materialCount, &increment, NULL, "%d"))
            {
                materialCount = std::max(0, std::min(materialCount, 8));

                meshRenderer->mMaterialCount = materialCount;
            }
            ImGui::PopItemWidth();

            Guid materialIds[8];
            for (int i = 0; i < materialCount; i++)
            {
                materialIds[i] = meshRenderer->getMaterial(i);

                Material* material = clipboard.getWorld()->getAssetByGuid<Material>(materialIds[i]);

                ImGui::SlotData data;
                if (ImGui::Slot2("Material", materialIds[i].isValid() ? material->getName()/*materialIds[i].toString()*/ : "None (Material)", &data))
                {
                    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Material)
                    {
                        materialIds[i] = clipboard.getDraggedId();
                        clipboard.clearDraggedItem();

                        meshRenderer->setMaterial(materialIds[i], i);
                    }

                    if (data.isClicked && materialIds[i].isValid())
                    {
                        clipboard.setSelectedItem(InteractionType::Material, materialIds[i]);
                    }

                    if (data.clearClicked)
                    {
                        meshRenderer->setMaterial(Guid::INVALID, i);
                    }
                }
            }

            bool isStatic = meshRenderer->mIsStatic;
            if (ImGui::Checkbox("Is Static?", &isStatic))
            {
                meshRenderer->mIsStatic = isStatic;
            }

            bool enabled = meshRenderer->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                meshRenderer->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}