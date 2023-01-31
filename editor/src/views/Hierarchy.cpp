#include <algorithm>

#include "../../include/views/Hierarchy.h"
#include "../../include/ProjectDatabase.h"
#include "../../include/imgui/imgui_extensions.h"
#include "imgui.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy() : Window("Hierarchy")
{
   
}

Hierarchy::~Hierarchy()
{
}

void Hierarchy::init(Clipboard &clipboard)
{
}

void Hierarchy::update(Clipboard &clipboard)
{
    clipboard.mOpen[static_cast<int>(View::Hierarchy)] = isOpen();
    clipboard.mHovered[static_cast<int>(View::Hierarchy)] = isHovered();
    clipboard.mFocused[static_cast<int>(View::Hierarchy)] = isFocused();
    clipboard.mOpenedThisFrame[static_cast<int>(View::Hierarchy)] = openedThisFrame();
    clipboard.mHoveredThisFrame[static_cast<int>(View::Hierarchy)] = hoveredThisFrame();
    clipboard.mFocusedThisFrame[static_cast<int>(View::Hierarchy)] = focusedThisFrame();
    clipboard.mClosedThisFrame[static_cast<int>(View::Hierarchy)] = closedThisFrame();
    clipboard.mUnfocusedThisFrame[static_cast<int>(View::Hierarchy)] = unfocusedThisFrame();
    clipboard.mUnhoveredThisFrame[static_cast<int>(View::Hierarchy)] = unhoveredThisFrame();

    if (clipboard.mSceneOpened)
    {
        // If number of entities has changed, update cached entity ids and names
        if (mEntries.size() != clipboard.getWorld()->getActiveScene()->getNumberOfNonHiddenEntities())
        {
            rebuildEntityLists(clipboard.getWorld());
        }

        if (clipboard.getSceneId().isValid())
        {
            // Set selected entity in hierarchy
            int selectedIndex;
            if (clipboard.getSelectedType() == InteractionType::Entity)
            {
                selectedIndex = mIdToEntryIndex[clipboard.getSelectedId()];
            }
            else
            {
                selectedIndex = -1;
            }

            // Check if scene is dirty and mark accordingly
            if (clipboard.mSceneDirty)
            {
                ImGui::Text((clipboard.getSceneName() + "*").c_str());
            }
            else
            {
                ImGui::Text(clipboard.getSceneName().c_str());
            }
            ImGui::Separator();

            // Display entities in hierarchy
            for (size_t i = 0; i < mEntries.size(); i++)
            {
                static bool selected = false;
                char buf1[64];
                std::size_t len = std::min(size_t(64 - 1), mEntries[i].entity->getName().length());
                strncpy(buf1, mEntries[i].entity->getName().c_str(), len);
                buf1[len] = '\0';

                bool edited = false;
                if (ImGui::SelectableInput(mEntries[i].label.c_str(), selectedIndex == i, &edited,
                    ImGuiSelectableFlags_DrawHoveredWhenHeld, buf1, IM_ARRAYSIZE(buf1)))
                {
                    mEntries[i].entity->setName(std::string(buf1));

                    clipboard.setSelectedItem(InteractionType::Entity, mEntries[i].entity->getGuid());
                }

                if (edited)
                {
                    clipboard.mSceneDirty = true;
                }

                if (ImGui::BeginDragDropSource())
                {
                    const void* data = static_cast<const void*>(mEntries[i].entity->getGuid().c_str());

                    ImGui::SetDragDropPayload("ENTITY_GUID", data, sizeof(PhysicsEngine::Guid));
                    ImGui::Text(mEntries[i].entity->getName().c_str());
                    ImGui::EndDragDropSource();
                }
            }

            // Right click popup menu
            if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
            {
                if (ImGui::MenuItem("Copy", NULL, false, clipboard.getSelectedType() == InteractionType::Entity))
                {
                }
                if (ImGui::MenuItem("Paste", NULL, false, clipboard.getSelectedType() == InteractionType::Entity))
                {
                }
                if (ImGui::MenuItem("Delete", NULL, false, clipboard.getSelectedType() == InteractionType::Entity) &&
                    clipboard.getSelectedType() == InteractionType::Entity)
                {
                    clipboard.getWorld()->getActiveScene()->latentDestroyEntity(clipboard.getSelectedId());

                    clipboard.clearSelectedItem();
                }

                ImGui::Separator();

                if (ImGui::BeginMenu("Create..."))
                {
                    if (ImGui::MenuItem("Empty"))
                    {
                        clipboard.getWorld()->getActiveScene()->createEntity();
                    }
                    if (ImGui::MenuItem("Camera"))
                    {
                        clipboard.getWorld()->getActiveScene()->createCamera();
                    }

                    if (ImGui::BeginMenu("Light"))
                    {
                        if (ImGui::MenuItem("Directional"))
                        {
                            clipboard.getWorld()->getActiveScene()->createLight(PhysicsEngine::LightType::Directional);
                        }
                        if (ImGui::MenuItem("Spot"))
                        {
                            clipboard.getWorld()->getActiveScene()->createLight(PhysicsEngine::LightType::Spot);
                        }
                        if (ImGui::MenuItem("Point"))
                        {
                            clipboard.getWorld()->getActiveScene()->createLight(PhysicsEngine::LightType::Point);
                        }
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("2D"))
                    {
                        if (ImGui::MenuItem("Plane"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Plane);
                        }
                        if (ImGui::MenuItem("Disc"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Disc);
                        }
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("3D"))
                    {
                        if (ImGui::MenuItem("Cube"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Cube);
                        }
                        if (ImGui::MenuItem("Sphere"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Sphere);
                        }
                        if (ImGui::MenuItem("Cylinder"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Cylinder);
                        }
                        if (ImGui::MenuItem("Cone"))
                        {
                            clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Cone);
                        }
                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndPopup();
            }

            // dropping mesh into hierarchy
            if (ImGui::BeginDragDropTarget())
            {
                const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MESH_PATH");
                if (payload != nullptr)
                {
                    const char* data = static_cast<const char*>(payload->Data);
                    std::filesystem::path incomingPath = std::string(data);

                    PhysicsEngine::Mesh* mesh = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Mesh>(ProjectDatabase::getGuid(incomingPath));
                    if (mesh != nullptr)
                    {
                        clipboard.getWorld()->getActiveScene()->createNonPrimitive(ProjectDatabase::getGuid(incomingPath));
                    }
                }

                ImGui::EndDragDropTarget();
            }




            /*if (ImGui::IsMouseReleased(0) && isHovered())
            {
                if (clipboard.getDraggedType() == InteractionType::Mesh)
                {
                    clipboard.getWorld()->getActiveScene()->createNonPrimitive(clipboard.getDraggedId());
                    clipboard.clearDraggedItem();
                }
            }*/
        }
    }
}

void Hierarchy::rebuildEntityLists(PhysicsEngine::World *world)
{
    mEntries.resize(world->getActiveScene()->getNumberOfNonHiddenEntities());

    int index = 0;
    for (size_t i = 0; i < world->getActiveScene()->getNumberOfEntities(); i++)
    {
        PhysicsEngine::Entity* entity = world->getActiveScene()->getEntityByIndex(i);
        if (entity->mHide == PhysicsEngine::HideFlag::None)
        {
            mEntries[index].entity = entity;
            mEntries[index].label = entity->getGuid().toString();
            mEntries[index].indentLevel = 0;
            mIdToEntryIndex[entity->getGuid()] = index;
            index++;
        }
    }
}