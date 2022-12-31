#include "../../include/drawers/TerrainDrawer.h"

#include "components/Terrain.h"

#include "imgui.h"
#include "imgui_internal.h"
//#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEditor;

TerrainDrawer::TerrainDrawer()
{
    //Renderer::getRenderer()->createFramebuffer(256, 256, &mFBO, &mColor);
    mFBO = Framebuffer::create(256, 256, 1, false);

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

    mProgram = ShaderProgram::create();
    mProgram->load("TerrainDrawer", vertexShader, fragmentShader);
    mProgram->compile();
    //ShaderStatus status;
    //Renderer::getRenderer()->compile("TerrainDrawer", vertexShader, fragmentShader, "", &mProgram, status);
}

TerrainDrawer::~TerrainDrawer()
{
    delete mFBO;
    delete mProgram;
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
                /*Guid transformId = terrain->mCameraTransformId;

                Entity* entity = nullptr;
                if (transformId.isValid())
                {
                    entity = clipboard.getWorld()->getActiveScene()->getComponentByGuid<Transform>(transformId)->getEntity();
                }*/

                ImVec2 windowSize = ImGui::GetWindowSize();
                windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

                if (ImGui::ButtonEx((transform == nullptr ? "None (Transform)" : transform->getEntity()->getName()).c_str(), ImVec2(windowSize.x, 0)))
                {
                }

                if (ImGui::BeginDragDropTarget())
                {
                    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ENTITY");
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





                /*ImGui::SlotData data;
                if (ImGui::Slot2("Transform", transformId.isValid() ? entity->getName() : "None (Transform)", &data))
                {
                    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Entity)
                    {
                        Guid entityId = clipboard.getDraggedId();
                        clipboard.clearDraggedItem();

                        terrain->mCameraTransformId = clipboard.getWorld()->getActiveScene()->getComponent<Transform>(entityId)->getGuid();
                    }

                    if (data.clearClicked)
                    {
                        terrain->mCameraTransformId = Guid::INVALID;
                    }
                }*/
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
                    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MATERIAL");
                    if (payload != nullptr)
                    {
                        const PhysicsEngine::Guid* data = static_cast<const PhysicsEngine::Guid*>(payload->Data);

                        terrain->setMaterial(*data);
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



                //ImGui::SlotData data;
                //if (ImGui::Slot2("Material", materialId.isValid() ? material->getName()/*materialId.toString()*/ : "None (Material)", &data))
                //{
                //    if (data.releaseTriggered && clipboard.getDraggedType() == InteractionType::Material)
                //    {
                //        materialId = clipboard.getDraggedId();
                //        clipboard.clearDraggedItem();

                //        terrain->setMaterial(materialId);
                //    }

                //    if (data.isClicked && materialId.isValid())
                //    {
                //        clipboard.setSelectedItem(InteractionType::Material, materialId);
                //    }

                //    if (data.clearClicked)
                //    {
                //        terrain->setMaterial(Guid::INVALID);
                //    }
                //}
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

            //Renderer::getRenderer()->bindFramebuffer(mFBO);
            //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
            //Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
            mFBO->bind();
            mFBO->setViewport(0, 0, 256, 256);
            mFBO->clearColor(Color::black);

            //Renderer::getRenderer()->use(mProgram);
            mProgram->bind();
            Renderer::getRenderer()->render(0, terrain->getVertices().size() / 3, terrain->getNativeGraphicsVAO());
            //Renderer::getRenderer()->unuse();
            //Renderer::getRenderer()->unbindFramebuffer();
            mProgram->unbind();
            mFBO->unbind();

            /*ImGui::Image((void*)(intptr_t)mColor,
                ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                ImVec2(0, 0));*/
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