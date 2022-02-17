#include "../../include/drawers/TerrainDrawer.h"

#include "components/Terrain.h"

#include "imgui.h"
#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

TerrainDrawer::TerrainDrawer()
{
    Graphics::createFramebuffer(256, 256, &mFBO, &mColor);

    std::string vertexShader = "#version 430 core\n"
        "in vec3 position;\n"
        "out float height;\n"
        "void main()\n"
        "{\n"
        "   height = position.y;\n"
        "	gl_Position = vec4(position, 1.0);\n"
        "}";

    std::string fragmentShader = "#version 430 core\n"
        "out vec4 FragColor;\n"
        "in float height;\n"
        "void main()\n"
        "{\n"
        "    FragColor = vec4(height, height, height, 1);\n"
        "}";

    Graphics::compile("TerrainDrawer", vertexShader, fragmentShader, "", &mProgram);
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
            {
                Guid transformId = terrain->mCameraTransformId;

                ImGui::SlotData data;
                if (ImGui::Slot2("Transform", transformId.isValid() ? transformId.toString() : "None (Transform)", &data))
                {
                    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Entity)
                    {
                        Guid entityId = clipboard.getDraggedId();
                        clipboard.clearDraggedItem();

                        terrain->mCameraTransformId = clipboard.getWorld()->getComponent<Transform>(entityId)->getId();
                    }

                    if (data.clearClicked)
                    {
                        terrain->mCameraTransformId = Guid::INVALID;
                    }
                }
            }









            //static bool generateModeActive = true;
            //static bool grassModeActive = false;
            //static bool treeModeActive = false;

            /*if (ImGui::StampButton("T", translationModeActive))
            {
                translationModeActive = true;
                rotationModeActive = false;
                scaleModeActive = false;
                operation = ImGuizmo::OPERATION::TRANSLATE;
            }
            ImGui::SameLine();

            if (ImGui::StampButton("R", rotationModeActive))
            {
                translationModeActive = false;
                rotationModeActive = true;
                scaleModeActive = false;
                operation = ImGuizmo::OPERATION::ROTATE;
            }
            ImGui::SameLine();

            if (ImGui::StampButton("S", scaleModeActive))
            {
                translationModeActive = false;
                rotationModeActive = false;
                scaleModeActive = true;
                operation = ImGuizmo::OPERATION::SCALE;
            }*/


















            // Material
            {
                Guid materialId = terrain->getMaterial();

                ImGui::SlotData data;
                if (ImGui::Slot2("Material", materialId.isValid() ? materialId.toString() : "None (Material)", &data))
                {
                    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Material)
                    {
                        materialId = clipboard.getDraggedId();
                        clipboard.clearDraggedItem();

                        terrain->setMaterial(materialId);
                    }

                    if (data.isClicked && materialId.isValid())
                    {
                        clipboard.setSelectedItem(InteractionType::Material, materialId);
                    }

                    if (data.clearClicked)
                    {
                        terrain->setMaterial(Guid::INVALID);
                    }
                }
            }

            float maxViewDistance = terrain->mMaxViewDistance;
            if (ImGui::InputFloat("Max View Dist", &maxViewDistance))
            {
                terrain->mMaxViewDistance = maxViewDistance;
            
                terrain->regenerateTerrain();
            }

            if (ImGui::SliderFloat("Scale", &terrain->mScale, 0.0f, 1.0f))
            {
                terrain->updateTerrainHeight();
            }

            if (ImGui::SliderFloat("Amplitude", &terrain->mAmplitude, 0.0f, 10.0f))
            {
                terrain->updateTerrainHeight();
            }

            if (ImGui::SliderFloat("OffsetX", &terrain->mOffsetX, 0.0f, 10.0f))
            {
                terrain->updateTerrainHeight();
            }

            if (ImGui::SliderFloat("OffsetZ", &terrain->mOffsetZ, 0.0f, 10.0f))
            {
                terrain->updateTerrainHeight();
            }

            Graphics::bindFramebuffer(mFBO);
            Graphics::setViewport(0, 0, 256, 256);
            Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
            Graphics::use(mProgram);
            Graphics::render(0, terrain->getVertices().size() / 3, terrain->getNativeGraphicsVAO());
            Graphics::unuse();
            Graphics::unbindFramebuffer();

            /*ImGui::Image((void*)(intptr_t)mColor,
                ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                ImVec2(0, 0));*/
            ImGui::Image((void*)(intptr_t)mColor,
                ImVec2(std::min(ImGui::GetWindowContentRegionWidth(), 256.0f), 256), ImVec2(1, 1),
                ImVec2(0, 0));

            if (ImGui::TreeNodeEx("Grass", ImGuiTreeNodeFlags_DefaultOpen))
            {
                // Grass Meshes
                int grassMeshCount = terrain->mGrassMeshCount;
                const int increment = 1;
                ImGui::PushItemWidth(80);
                if (ImGui::InputScalar("Mesh Count", ImGuiDataType_S32, &grassMeshCount, &increment, NULL, "%d"))
                {
                    grassMeshCount = std::max(0, std::min(grassMeshCount, 8));

                    terrain->mGrassMeshCount = grassMeshCount;
                }
                ImGui::PopItemWidth();

                Guid grassMeshIds[8];
                for (int i = 0; i < grassMeshCount; i++)
                {
                    grassMeshIds[i] = terrain->getGrassMesh(i);

                    ImGui::SlotData data;
                    if (ImGui::Slot2("Mesh", grassMeshIds[i].isValid() ? grassMeshIds[i].toString() : "None (Mesh)", &data))
                    {
                        if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Mesh)
                        {
                            grassMeshIds[i] = clipboard.getDraggedId();
                            clipboard.clearDraggedItem();

                            terrain->setGrassMesh(grassMeshIds[i], i);
                        }

                        if (data.isClicked && grassMeshIds[i].isValid())
                        {
                            clipboard.setSelectedItem(InteractionType::Mesh, grassMeshIds[i]);
                        }

                        if (data.clearClicked)
                        {
                            terrain->setGrassMesh(Guid::INVALID, i);
                        }
                    }




                    


                }

                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Trees", ImGuiTreeNodeFlags_DefaultOpen))
            {
                // Tree Meshes
                int treeMeshCount = terrain->mTreeMeshCount;
                const int increment = 1;
                ImGui::PushItemWidth(80);
                if (ImGui::InputScalar("Mesh Count", ImGuiDataType_S32, &treeMeshCount, &increment, NULL, "%d"))
                {
                    treeMeshCount = std::max(0, std::min(treeMeshCount, 8));

                    terrain->mTreeMeshCount = treeMeshCount;
                }
                ImGui::PopItemWidth();

                Guid treeMeshIds[8];
                for (int i = 0; i < treeMeshCount; i++)
                {
                    treeMeshIds[i] = terrain->getTreeMesh(i);

                    ImGui::SlotData data;
                    if (ImGui::Slot2("Mesh", treeMeshIds[i].isValid() ? treeMeshIds[i].toString() : "None (Mesh)", &data))
                    {
                        if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Mesh)
                        {
                            treeMeshIds[i] = clipboard.getDraggedId();
                            clipboard.clearDraggedItem();

                            terrain->setTreeMesh(treeMeshIds[i], i);
                        }

                        if (data.isClicked && treeMeshIds[i].isValid())
                        {
                            clipboard.setSelectedItem(InteractionType::Mesh, treeMeshIds[i]);
                        }

                        if (data.clearClicked)
                        {
                            terrain->setTreeMesh(Guid::INVALID, i);
                        }
                    }
                }

                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}