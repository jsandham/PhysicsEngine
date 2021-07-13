#include "../../include/views/Inspector.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/FileSystemUtil.h"
#include "../../include/drawers/LoadInspectorDrawerInternal.h"

#include "imgui.h"

#include "../include/components/Light.h"
#include "../include/components/MeshRenderer.h"

using namespace PhysicsEditor;

Inspector::Inspector() : Window("Inspector")
{
}

Inspector::~Inspector()
{
}

void Inspector::init(Clipboard &clipboard)
{
}

void Inspector::update(Clipboard &clipboard)
{
    // draw selected asset
    if (clipboard.getSelectedType() == InteractionType::Mesh)
    {
        mMeshDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Material)
    {
        mMaterialDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Shader)
    {
        mShaderDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Texture2D)
    {
        mTexture2DDrawer.render(clipboard, clipboard.getSelectedId());
    }
    else if (clipboard.getSelectedType() == InteractionType::Sprite)
    {
        mSpriteDrawer.render(clipboard, clipboard.getSelectedId());
    }

    // draw selected entity
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        drawEntity(clipboard);
    }

    ImGui::Separator();
}

void Inspector::drawEntity(Clipboard &clipboard)
{
    Entity *entity = clipboard.getWorld()->getEntityById(clipboard.getSelectedId());

    // entity may have been recently deleted
    if (entity == nullptr)
    {
        return;
    }

    std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(clipboard.getWorld());
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        Guid componentId = componentsOnEntity[i].first;
        int componentType = componentsOnEntity[i].second;

        InspectorDrawer *drawer = nullptr;
        if (Component::isInternal(componentType))
        {
            if (componentType == ComponentType<Transform>::type)
            {
                drawer = &mTransformDrawer;
            }
            else if (componentType == ComponentType<Rigidbody>::type)
            {
                drawer = &mRigidbodyDrawer;
            }
            else if (componentType == ComponentType<Camera>::type)
            {
                drawer = &mCameraDrawer;
            }
            else if (componentType == ComponentType<MeshRenderer>::type)
            {
                drawer = &mMeshRendererDrawer;
            }
            else if (componentType == ComponentType<SpriteRenderer>::type)
            {
                drawer = &mSpriteRendererDrawer;
            }
            else if (componentType == ComponentType<LineRenderer>::type)
            {
                drawer = &mLineRendererDrawer;
            }
            else if (componentType == ComponentType<Light>::type)
            {
                drawer = &mLightDrawer;
            }
            else if (componentType == ComponentType<BoxCollider>::type)
            {
                drawer = &mBoxColliderDrawer;
            }
            else if (componentType == ComponentType<SphereCollider>::type)
            {
                drawer = &mSphereColliderDrawer;
            }
            else if (componentType == ComponentType<CapsuleCollider>::type)
            {
                drawer = &mCapsuleColliderDrawer;
            }
            else if (componentType == ComponentType<MeshCollider>::type)
            {
                drawer = &mMeshColliderDrawer;
            }
        }

        drawer->render(clipboard, componentId);
        ImGui::Separator();
    }

    std::string componentToAdd = "";
    std::vector<std::string> components = {"Transform",    "Camera",       "Light",       "Rigidbody",
                                           "MeshRenderer", "SpriteRenderer", "LineRenderer", "BoxCollider", 
                                           "SphereCollider"};

    if (ImGui::BeginDropdownWindow("Add component", components, componentToAdd))
    {
        if (componentToAdd == "Transform")
        {
            Undo::addCommand(
                new AddComponentCommand<Transform>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }
        else if (componentToAdd == "Rigidbody")
        {
            Undo::addCommand(
                new AddComponentCommand<Rigidbody>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }
        else if (componentToAdd == "Camera")
        {
            Undo::addCommand(
                new AddComponentCommand<Camera>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }
        else if (componentToAdd == "MeshRenderer")
        {
            Undo::addCommand(
                new AddComponentCommand<MeshRenderer>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }
        else if (componentToAdd == "SpriteRenderer")
        {
            Undo::addCommand(
                new AddComponentCommand<SpriteRenderer>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }
        else if (componentToAdd == "Light")
        {
            Undo::addCommand(
                new AddComponentCommand<Light>(clipboard.getWorld(), entity->getId(), &clipboard.mSceneDirty));
        }

        ImGui::EndDropdownWindow();
    }
}