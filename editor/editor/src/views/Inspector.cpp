#include "../../include/views/Inspector.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/FileSystemUtil.h"
#include "../../include/drawers/LoadInspectorDrawerInternal.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../include/components/Light.h"
#include "../include/components/MeshRenderer.h"

using namespace PhysicsEditor;

Inspector::Inspector() : Window("Inspector")
{
}

Inspector::~Inspector()
{
}

void Inspector::init(EditorClipboard &clipboard)
{
}

void Inspector::update(EditorClipboard &clipboard)
{
    // draw selected entity
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        drawEntity(clipboard);
    }

    // draw selected asset
    if (clipboard.getSelectedType() == InteractionType::Mesh)
    {
        meshDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Material)
    {
        materialDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Shader)
    {
        shaderDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Texture2D)
    {
        texture2DDrawer.render(clipboard, clipboard.getSelectedId());
    }

    ImGui::Separator();
}

void Inspector::drawEntity(EditorClipboard &clipboard)
{
    Entity *entity = clipboard.getWorld()->getEntityById(clipboard.getSelectedId());

    // entity may have been recently deleted
    if (entity == NULL)
    {
        return;
    }

    std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(clipboard.getWorld());
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

        drawer->render(clipboard, componentId);
        ImGui::Separator();

        delete drawer;
    }

    std::string componentToAdd = "";
    std::vector<std::string> components = {"Transform",    "Camera",       "Light",       "Rigidbody",
                                           "MeshRenderer", "LineRenderer", "BoxCollider", "SphereCollider"};

    if (ImGui::BeginDropdownWindow("Add component", components, componentToAdd))
    {

        if (componentToAdd == "Transform")
        {
            Undo::addCommand(
                new AddComponentCommand<Transform>(clipboard.getWorld(), entity->getId(), &clipboard.isDirty));
        }
        else if (componentToAdd == "Rigidbody")
        {
            Undo::addCommand(
                new AddComponentCommand<Rigidbody>(clipboard.getWorld(), entity->getId(), &clipboard.isDirty));
        }
        else if (componentToAdd == "Camera")
        {
            Undo::addCommand(
                new AddComponentCommand<Camera>(clipboard.getWorld(), entity->getId(), &clipboard.isDirty));
        }
        else if (componentToAdd == "MeshRenderer")
        {
            Undo::addCommand(
                new AddComponentCommand<MeshRenderer>(clipboard.getWorld(), entity->getId(), &clipboard.isDirty));
        }
        else if (componentToAdd == "Light")
        {
            Undo::addCommand(
                new AddComponentCommand<Light>(clipboard.getWorld(), entity->getId(), &clipboard.isDirty));
        }

        ImGui::EndDropdownWindow();
    }
}