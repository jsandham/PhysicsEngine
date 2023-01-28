#include "../../include/drawers/TerrainDrawer.h"
#include "../../include/ProjectDatabase.h"

#include "components/Terrain.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

TerrainDrawer::TerrainDrawer()
{
    mFBO = Framebuffer::create(256, 256, 1, false);
}

TerrainDrawer::~TerrainDrawer()
{
    delete mFBO;
}

void TerrainDrawer::render(Clipboard& clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Terrain", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Terrain* terrain = clipboard.getWorld()->getActiveScene()->getComponentByGuid<Terrain>(id);

        if (terrain != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            // Transform
            {
                Transform* transform = clipboard.getWorld()->getActiveScene()->getComponentByGuid<Transform>(terrain->mCameraTransformId);

                ImVec2 windowSize = ImGui::GetWindowSize();
                windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

                if (ImGui::ButtonEx((transform == nullptr ? "None (Transform)" : transform->getEntity()->getName()).c_str(), ImVec2(windowSize.x, 0)))
                {
                }

                if (ImGui::BeginDragDropTarget())
                {
                    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ENTITY_GUID");
                    if (payload != nullptr)
                    {
                        const PhysicsEngine::Guid* data = static_cast<const PhysicsEngine::Guid*>(payload->Data);

                        terrain->mCameraTransformId = clipboard.getWorld()->getActiveScene()->getComponent<Transform>(*data)->getGuid();
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
                ImGui::Text("Transform");
            }

            // Material
            {
                Material* material = clipboard.getWorld()->getAssetByGuid<Material>(terrain->getMaterial());

                ImVec2 windowSize = ImGui::GetWindowSize();
                windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

                if (ImGui::ButtonEx((material == nullptr ? "None (Material)" : material->getName()).c_str(), ImVec2(windowSize.x, 0)))
                {
                    clipboard.setSelectedItem(InteractionType::Material, material->getGuid());
                }

                if (ImGui::BeginDragDropTarget())
                {
                    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MATERIAL_PATH");
                    if (payload != nullptr)
                    {
                        const char* data = static_cast<const char*>(payload->Data);

                        terrain->setMaterial(ProjectDatabase::getGuid(data));
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
                ImGui::Text("Material");
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

            Shader* shader = clipboard.getWorld()->getAssetByGuid<Shader>(Guid("336b168c-3b92-473d-909a-0a2e342d483f"));
            
            assert(shader != nullptr);
            
            mProgram = shader->getProgramFromVariant(0);

            mFBO->bind();
            mFBO->setViewport(0, 0, 256, 256);
            mFBO->clearColor(Color::black);

            mProgram->bind();
            Renderer::getRenderer()->render(0, terrain->getVertices().size() / 3, terrain->getNativeGraphicsVAO());
            mProgram->unbind();
            mFBO->unbind();

            ImGui::Image((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle())),
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

                //Guid grassMeshIds[8];
                //for (int i = 0; i < grassMeshCount; i++)
                //{
                //    grassMeshIds[i] = terrain->getGrassMesh(i);

                //    Mesh* grassMesh = clipboard.getWorld()->getAssetByGuid<Mesh>(grassMeshIds[i]);

                //    ImGui::SlotData data;
                //    if (ImGui::Slot2("Mesh", grassMeshIds[i].isValid() ? grassMesh->getName()/*grassMeshIds[i].toString()*/ : "None (Mesh)", &data))
                //    {
                //        if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Mesh)
                //        {
                //            grassMeshIds[i] = clipboard.getDraggedId();
                //            clipboard.clearDraggedItem();

                //            terrain->setGrassMesh(grassMeshIds[i], i);
                //        }

                //        if (data.isClicked && grassMeshIds[i].isValid())
                //        {
                //            clipboard.setSelectedItem(InteractionType::Mesh, grassMeshIds[i]);
                //        }

                //        if (data.clearClicked)
                //        {
                //            terrain->setGrassMesh(Guid::INVALID, i);
                //        }
                //    }
                //}

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

                //Guid treeMeshIds[8];
                //for (int i = 0; i < treeMeshCount; i++)
                //{
                //    treeMeshIds[i] = terrain->getTreeMesh(i);

                //    Mesh* treeMesh = clipboard.getWorld()->getAssetByGuid<Mesh>(treeMeshIds[i]);

                //    ImGui::SlotData data;
                //    if (ImGui::Slot2("Mesh", treeMeshIds[i].isValid() ? treeMesh->getName()/*treeMeshIds[i].toString()*/ : "None (Mesh)", &data))
                //    {
                //        if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Mesh)
                //        {
                //            treeMeshIds[i] = clipboard.getDraggedId();
                //            clipboard.clearDraggedItem();

                //            terrain->setTreeMesh(treeMeshIds[i], i);
                //        }

                //        if (data.isClicked && treeMeshIds[i].isValid())
                //        {
                //            clipboard.setSelectedItem(InteractionType::Mesh, treeMeshIds[i]);
                //        }

                //        if (data.clearClicked)
                //        {
                //            terrain->setTreeMesh(Guid::INVALID, i);
                //        }
                //    }
                //}

                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}