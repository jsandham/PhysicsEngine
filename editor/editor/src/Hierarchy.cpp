#include <algorithm>

#include "../include/Hierarchy.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"
#include "../include/imgui_extensions.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy()
{
}

Hierarchy::~Hierarchy()
{
}

void Hierarchy::render(World *world, EditorScene &scene, EditorClipboard &clipboard,
                       const std::set<Guid> &editorOnlyEntityIds, bool isOpenedThisFrame)
{
    static bool hierarchyActive = true;

    if (isOpenedThisFrame)
    {
        hierarchyActive = true;
    }

    if (!hierarchyActive)
    {
        return;
    }

    if (ImGui::Begin("Hierarchy", &hierarchyActive))
    {
        if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
        {
            ImGui::SetWindowFocus("Hierarchy");
        }

        rebuildRequired = entries.size() != std::max(0, world->getNumberOfEntities() - (int)editorOnlyEntityIds.size());

        // If number of entities has changed, update cached entity ids and names
        if (rebuildRequired)
        {
            rebuildEntityLists(world, editorOnlyEntityIds);
        }

        if (scene.name.length() > 0)
        {
            // Set selected entity in hierarchy
            int selectedIndex;
            if (clipboard.getSelectedType() == InteractionType::Entity) {
                selectedIndex = idToEntryIndex[clipboard.getSelectedId()];
            }
            else {
                selectedIndex = -1;
            }

            // Check if scene is dirty and mark accordingly
            if (scene.isDirty)
            {
                ImGui::Text((scene.name + "*").c_str());
            }
            else
            {
                ImGui::Text(scene.name.c_str());
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
                if (ImGui::SelectableInput(entries[i].label.c_str(), selectedIndex == i, &edited, ImGuiSelectableFlags_DrawHoveredWhenHeld, buf1, IM_ARRAYSIZE(buf1)))
                {
                    entries[i].entity->setName(std::string(buf1));

                    clipboard.setSelectedItem(InteractionType::Entity, entries[i].entity->getId());
                }

                if (edited) {
                    scene.isDirty = true;
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
                    world->latentDestroyEntity(clipboard.getSelectedId());

                    clipboard.clearSelectedItem();
                }

                ImGui::Separator();

                if (ImGui::BeginMenu("Create..."))
                {
                    if (ImGui::MenuItem("Empty"))
                    {
                        CommandManager::addCommand(new CreateEntityCommand(world, &scene.isDirty));
                    }
                    if (ImGui::MenuItem("Camera"))
                    {
                        CommandManager::addCommand(new CreateCameraCommand(world, &scene.isDirty));
                    }
                    if (ImGui::MenuItem("Light"))
                    {
                        CommandManager::addCommand(new CreateLightCommand(world, &scene.isDirty));
                    }

                    if (ImGui::BeginMenu("2D"))
                    {
                        if (ImGui::MenuItem("Plane"))
                        {
                            CommandManager::addCommand(new CreatePlaneCommand(world, &scene.isDirty));
                        }
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("3D"))
                    {
                        if (ImGui::MenuItem("Cube"))
                        {
                            CommandManager::addCommand(new CreateCubeCommand(world, &scene.isDirty));
                        }
                        if (ImGui::MenuItem("Sphere"))
                        {
                            CommandManager::addCommand(new CreateSphereCommand(world, &scene.isDirty));
                        }
                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndPopup();
            }
        }
    }

    ImGui::End();
}

void Hierarchy::rebuildEntityLists(World* world, const std::set<Guid>& editorOnlyEntityIds)
{
    int numberOfEntities = std::max(0, world->getNumberOfEntities() - (int)editorOnlyEntityIds.size());

    entries.resize(numberOfEntities);

    int index = 0;
    for (int i = 0; i < world->getNumberOfEntities(); i++)
    {
        Entity* entity = world->getEntityByIndex(i);

        std::set<Guid>::iterator it = editorOnlyEntityIds.find(entity->getId());
        if (it == editorOnlyEntityIds.end())
        {
            entries[index].entity = entity;
            entries[index].label = entity->getId().toString();
            entries[index].indentLevel = 0;
            idToEntryIndex[entity->getId()] = index;
            index++;
        }
    }
}