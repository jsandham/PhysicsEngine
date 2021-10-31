#include <algorithm>

#include "../../include/views/Hierarchy.h"

#include "../../include/imgui/imgui_extensions.h"
#include "imgui.h"
#include <shlobj.h>
#include <shlwapi.h>
#include <objbase.h>

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
    // If number of entities has changed, update cached entity ids and names
    if (mEntries.size() != clipboard.getWorld()->getNumberOfNonHiddenEntities())
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

                clipboard.setSelectedItem(InteractionType::Entity, mEntries[i].entity->getId());
            }

            if (edited)
            {
                clipboard.mSceneDirty = true;
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
                clipboard.getWorld()->latentDestroyEntity(clipboard.getSelectedId());

                clipboard.clearSelectedItem();
            }

            ImGui::Separator();

            if (ImGui::BeginMenu("Create..."))
            {
                if (ImGui::MenuItem("Empty"))
                {
                    //Undo::addCommand(new CreateEntityCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }
                if (ImGui::MenuItem("Camera"))
                {
                    //Undo::addCommand(new CreateCameraCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }
                if (ImGui::MenuItem("Light"))
                {
                    //Undo::addCommand(new CreateLightCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }

                if (ImGui::BeginMenu("2D"))
                {
                    if (ImGui::MenuItem("Plane"))
                    {
                        //Undo::addCommand(
                        //    new CreatePlaneCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("3D"))
                {
                    if (ImGui::MenuItem("Cube"))
                    {
                        //Undo::addCommand(new CreateCubeCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    if (ImGui::MenuItem("Sphere"))
                    {
                        //Undo::addCommand(
                        //    new CreateSphereCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }

            ImGui::EndPopup();
        }

        // dropping mesh into hierarchy
        if (ImGui::IsMouseReleased(0) && isHovered())
        {
            if (clipboard.getDraggedType() == InteractionType::Mesh)
            {
                PhysicsEngine::Entity* entity = clipboard.getWorld()->createEntity();
                PhysicsEngine::Transform* transform = entity->addComponent<PhysicsEngine::Transform>();
                PhysicsEngine::MeshRenderer* meshRenderer = entity->addComponent<PhysicsEngine::MeshRenderer>();
                meshRenderer->setMesh(clipboard.getDraggedId());
                /*meshRenderer->setMaterial(clipboard.getWorld()->getColorMaterial());*/
                meshRenderer->setMaterial(clipboard.getWorld()->getAssetId("data\\materials\\default.material"));

                clipboard.clearDraggedItem();
            }
        }
    }
}

void Hierarchy::rebuildEntityLists(PhysicsEngine::World *world)
{
    mEntries.resize(world->getNumberOfNonHiddenEntities());

    size_t index = 0;
    for (size_t i = 0; i < world->getNumberOfEntities(); i++)
    {
        PhysicsEngine::Entity* entity = world->getEntityByIndex(i);
        if (entity->mHide == PhysicsEngine::HideFlag::None)
        {
            mEntries[index].entity = entity;
            mEntries[index].label = entity->getId().toString();
            mEntries[index].indentLevel = 0;
            mIdToEntryIndex[entity->getId()] = index;
            index++;
        }
    }
}