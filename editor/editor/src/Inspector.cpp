#include "../include/Inspector.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"
#include "../include/FileSystemUtil.h"
#include "../include/LoadInspectorDrawerInternal.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../include/components/Light.h"
#include "../include/components/MeshRenderer.h"

using namespace PhysicsEditor;

Inspector::Inspector()
{
}

Inspector::~Inspector()
{
}

void Inspector::render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                       bool isOpenedThisFrame)
{
    static bool inspectorActive = true;

    if (isOpenedThisFrame)
    {
        inspectorActive = true;
    }

    if (!inspectorActive)
    {
        return;
    }

    if (ImGui::Begin("Inspector", &inspectorActive))
    {
        if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
        {
            ImGui::SetWindowFocus("Inspector");
        }

        // draw selected entity
        if (clipboard.getSelectedType() == InteractionType::Entity)
        {
            drawEntity(world, project, scene, clipboard);
        }

        // draw selected asset
        if (clipboard.getSelectedType() == InteractionType::Mesh)
        {
            meshDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
        }
        else if (clipboard.getSelectedType() == InteractionType::Material)
        {
            materialDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
        }
        else if (clipboard.getSelectedType() == InteractionType::Shader)
        {
            shaderDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
        }
        else if (clipboard.getSelectedType() == InteractionType::Texture2D)
        {
            texture2DDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
        }

        ImGui::Separator();
    }

    ImGui::End();
}

void Inspector::drawEntity(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard)
{
    Entity *entity = world->getEntityById(clipboard.getSelectedId());

    // entity may have been recently deleted
    if (entity == NULL)
    {
        return;
    }

    std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        Guid componentId = componentsOnEntity[i].first;
        int componentType = componentsOnEntity[i].second;

        InspectorDrawer *drawer = NULL;
        if (Component::isInternal(componentType))
        {
            drawer = loadInternalInspectorComponentDrawer(componentType);
        }
        else
        {
            // drawer = loadInspectorDrawer(componentType);
        }

        drawer->render(world, project, scene, clipboard, componentId);
        ImGui::Separator();

        delete drawer;
    }

    std::string componentToAdd = "";
    std::vector<std::string> components = { "Transform",    "Camera",       "Light",       "Rigidbody",
                                            "MeshRenderer", "LineRenderer", "BoxCollider", "SphereCollider"};

    if (ImGui::BeginDropdownWindow("Add component", components, componentToAdd))
    {

        if (componentToAdd == "Transform")
        {
            CommandManager::addCommand(new AddComponentCommand<Transform>(world, entity->getId(), &scene.isDirty));
        }
        else if (componentToAdd == "Rigidbody")
        {
            CommandManager::addCommand(new AddComponentCommand<Rigidbody>(world, entity->getId(), &scene.isDirty));
        }
        else if (componentToAdd == "Camera")
        {
            CommandManager::addCommand(new AddComponentCommand<Camera>(world, entity->getId(), &scene.isDirty));
        }
        else if (componentToAdd == "MeshRenderer")
        {
            CommandManager::addCommand(new AddComponentCommand<MeshRenderer>(world, entity->getId(), &scene.isDirty));
        }
        else if (componentToAdd == "Light")
        {
            CommandManager::addCommand(new AddComponentCommand<Light>(world, entity->getId(), &scene.isDirty));
        }

        ImGui::EndDropdownWindow();
    }
}