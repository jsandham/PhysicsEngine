#include "../../include/drawers/TerrainDrawer.h"

#include "components/Terrain.h"

#include "imgui.h"
#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

TerrainDrawer::TerrainDrawer()
{
}

TerrainDrawer::~TerrainDrawer()
{
}

void TerrainDrawer::render(Clipboard& clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Terrain", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Terrain* terrain = clipboard.getWorld()->getComponentById<Terrain>(id);

        if (terrain != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            // Transform
            /*{
                Guid transformId = terrain->mCameraTransformId;

                std::string transformName = "None (Transform)";
                if (transformId.isValid())
                {
                    transformName = transformId.toString();
                }

                bool releaseTriggered = false;
                bool clearClicked = false;
                bool isClicked = ImGui::Slot("Transform", transformName, &releaseTriggered, &clearClicked);

                if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Entity)
                {
                    transformId = clipboard.getDraggedId();
                    clipboard.clearDraggedItem();

                    terrain->mCameraTransformId = transformId;
                }

                if (isClicked && transformId.isValid())
                {
                    clipboard.setSelectedItem(InteractionType::Entity, transformId);
                }
            }*/

            // Material
            {
                Guid materialId = terrain->getMaterial();

                std::string materialName = "None (Material)";
                if (materialId.isValid())
                {
                    materialName = materialId.toString();
                }

                bool releaseTriggered = false;
                bool clearClicked = false;
                bool isClicked = ImGui::Slot("Material", materialName, &releaseTriggered, &clearClicked);

                if (releaseTriggered && clipboard.getDraggedType() == InteractionType::Material)
                {
                    materialId = clipboard.getDraggedId();
                    clipboard.clearDraggedItem();

                    terrain->setMaterial(materialId);
                }

                if (isClicked && materialId.isValid())
                {
                    clipboard.setSelectedItem(InteractionType::Material, materialId);
                }
            }

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    std::string title = "##Chunk " + std::to_string(3 * i + j) + " enabled?";
                    bool enabled = terrain->isChunkEnabled(3 * i + j);
                    if (ImGui::Checkbox(title.c_str(), &enabled))
                    {
                        enabled ? terrain->enableChunk(3 * i + j) : terrain->disableChunk(3 * i + j);
                    }

                    if (j < 2)
                    {
                        ImGui::SameLine();
                    }
                }
            }

            if (ImGui::SliderFloat("Scale", &terrain->mScale, 0.0f, 2.0f))
            {
                terrain->regenerateTerrain();
            }

            if (ImGui::SliderFloat("Amplitude", &terrain->mAmplitude, 0.0f, 10.0f))
            {
                terrain->regenerateTerrain();
            }

            if (ImGui::SliderFloat("OffsetX", &terrain->mOffsetX, 0.0f, 10.0f))
            {
                terrain->regenerateTerrain();
            }

            if (ImGui::SliderFloat("OffsetZ", &terrain->mOffsetZ, 0.0f, 10.0f))
            {
                terrain->regenerateTerrain();
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}