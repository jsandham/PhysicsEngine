#include <algorithm>

#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/views/Hierarchy.h"

#include "../../include/imgui/imgui_extensions.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

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
    /*rebuildRequired = entries.size() != std::max((size_t)0, clipboard.getWorld()->getNumberOfEntities() -
        clipboard.getEditorOnlyIds().size());*/
    rebuildRequired = entries.size() != clipboard.getWorld()->getNumberOfNonHiddenEntities();

    // If number of entities has changed, update cached entity ids and names
    if (rebuildRequired)
    {
        rebuildEntityLists(clipboard.getWorld());
    }

    if (clipboard.getSceneId().isValid())
    {
        // Set selected entity in hierarchy
        int selectedIndex;
        if (clipboard.getSelectedType() == InteractionType::Entity)
        {
            selectedIndex = idToEntryIndex[clipboard.getSelectedId()];
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
        for (size_t i = 0; i < entries.size(); i++)
        {
            static bool selected = false;
            char buf1[64];
            std::size_t len = std::min(size_t(64 - 1), entries[i].entity->getName().length());
            strncpy(buf1, entries[i].entity->getName().c_str(), len);
            buf1[len] = '\0';

            bool edited = false;
            if (ImGui::SelectableInput(entries[i].label.c_str(), selectedIndex == i, &edited,
                ImGuiSelectableFlags_DrawHoveredWhenHeld, buf1, IM_ARRAYSIZE(buf1)))
            {
                entries[i].entity->setName(std::string(buf1));

                clipboard.setSelectedItem(InteractionType::Entity, entries[i].entity->getId());
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
                    Undo::addCommand(new CreateEntityCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }
                if (ImGui::MenuItem("Camera"))
                {
                    Undo::addCommand(new CreateCameraCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }
                if (ImGui::MenuItem("Light"))
                {
                    Undo::addCommand(new CreateLightCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                }

                if (ImGui::BeginMenu("2D"))
                {
                    if (ImGui::MenuItem("Plane"))
                    {
                        Undo::addCommand(
                            new CreatePlaneCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("3D"))
                {
                    if (ImGui::MenuItem("Cube"))
                    {
                        Undo::addCommand(new CreateCubeCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    if (ImGui::MenuItem("Sphere"))
                    {
                        Undo::addCommand(
                            new CreateSphereCommand(clipboard.getWorld(), &clipboard.mSceneDirty));
                    }
                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }

            ImGui::EndPopup();
        }
    }
}

void Hierarchy::rebuildEntityLists(World *world)
{
    entries.resize(world->getNumberOfNonHiddenEntities());

    size_t index = 0;
    for (size_t i = 0; i < world->getNumberOfEntities(); i++)
    {
        Entity* entity = world->getEntityByIndex(i);
        if (!entity->mHide)
        {
            entries[index].entity = entity;
            entries[index].label = entity->getId().toString();
            entries[index].indentLevel = 0;
            idToEntryIndex[entity->getId()] = index;
            index++;
        }
    }

    /*int numberOfEntities = std::max((size_t)0, world->getNumberOfEntities() - editorOnlyEntityIds.size());

    entries.resize(numberOfEntities);

    int index = 0;
    for (int i = 0; i < world->getNumberOfEntities(); i++)
    {
        Entity *entity = world->getEntityByIndex(i);

        std::set<Guid>::iterator it = editorOnlyEntityIds.find(entity->getId());
        if (it == editorOnlyEntityIds.end())
        {
            entries[index].entity = entity;
            entries[index].label = entity->getId().toString();
            entries[index].indentLevel = 0;
            idToEntryIndex[entity->getId()] = index;
            index++;
        }
    }*/
}