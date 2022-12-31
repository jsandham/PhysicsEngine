#include "../../include/views/Inspector.h"
#include "../../include/FileSystemUtil.h"

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
    clipboard.mOpen[static_cast<int>(View::Inspector)] = isOpen();
    clipboard.mHovered[static_cast<int>(View::Inspector)] = isHovered();
    clipboard.mFocused[static_cast<int>(View::Inspector)] = isFocused();
    clipboard.mOpenedThisFrame[static_cast<int>(View::Inspector)] = openedThisFrame();
    clipboard.mHoveredThisFrame[static_cast<int>(View::Inspector)] = hoveredThisFrame();
    clipboard.mFocusedThisFrame[static_cast<int>(View::Inspector)] = focusedThisFrame();
    clipboard.mClosedThisFrame[static_cast<int>(View::Inspector)] = closedThisFrame();
    clipboard.mUnfocusedThisFrame[static_cast<int>(View::Inspector)] = unfocusedThisFrame();
    clipboard.mUnhoveredThisFrame[static_cast<int>(View::Inspector)] = unhoveredThisFrame();

    // draw selected asset
    switch (clipboard.getSelectedType())
    {
    case InteractionType::Scene:
        mSceneDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Cubemap:
        mCubemapDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Mesh:
        mMeshDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Material:
        mMaterialDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Shader:
        mShaderDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Texture2D:
        mTexture2DDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::Sprite:
        mSpriteDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    case InteractionType::RenderTexture:
        mRenderTextureDrawer.render(clipboard, clipboard.getSelectedId());
        break;
    }

    // draw selected entity
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        drawEntity(clipboard);
    }
}

void Inspector::drawEntity(Clipboard &clipboard)
{
    Entity *entity = clipboard.getWorld()->getActiveScene()->getEntityByGuid(clipboard.getSelectedId());

    // entity may have been recently deleted
    if (entity == nullptr)
    {
        return;
    }

    static bool flag = true;
    ImGui::Checkbox("##Entity Enabled", &flag);
    
    ImGui::SameLine();
    ImGui::Text("Entity");
    ImGui::SameLine();

    std::string name = entity->getName();

    std::vector<char> inputBuffer(256, '\0');
    std::copy(name.begin(), name.end(), inputBuffer.begin());

    ImGuiInputTextFlags options = ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue;
    if (ImGui::InputText("##Entity Header", &inputBuffer[0], inputBuffer.size(), options))
    {
        entity->setName(std::string(inputBuffer.begin(), inputBuffer.end()));
    }

    ImGui::Text(("EntityId: " + entity->getGuid().toString()).c_str());

    ImGui::Separator();

    std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity();
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        Guid componentId = componentsOnEntity[i].first;
        int componentType = componentsOnEntity[i].second;

        ImGui::PushID(componentId.c_str());

        InspectorDrawer *drawer = nullptr;
        if (Component::isInternal(componentType))
        {
            switch (componentType)
            {
            case ComponentType<Transform>::type: { drawer = &mTransformDrawer; break; }
            case ComponentType<Rigidbody>::type: { drawer = &mRigidbodyDrawer; break; }
            case ComponentType<Camera>::type: { drawer = &mCameraDrawer; break; }
            case ComponentType<MeshRenderer>::type: { drawer = &mMeshRendererDrawer; break; }
            case ComponentType<SpriteRenderer>::type: { drawer = &mSpriteRendererDrawer; break; }
            case ComponentType<LineRenderer>::type: { drawer = &mLineRendererDrawer; break; }
            case ComponentType<Light>::type: { drawer = &mLightDrawer; break; }
            case ComponentType<BoxCollider>::type: { drawer = &mBoxColliderDrawer; break; }
            case ComponentType<SphereCollider>::type: { drawer = &mSphereColliderDrawer; break; }
            case ComponentType<CapsuleCollider>::type: { drawer = &mCapsuleColliderDrawer; break; }
            case ComponentType<MeshCollider>::type: { drawer = &mMeshColliderDrawer; break; }
            case ComponentType<Terrain>::type: { drawer = &mTerrainDrawer; break; }
            }
        }

        if (drawer != nullptr) {
            drawer->render(clipboard, componentId);

            if (drawer->isHovered())
            {
                if (componentType != ComponentType<Transform>::type)
                {
                    if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
                    {
                        if (ImGui::MenuItem("RemoveComponent", NULL, false, true))
                        {
                            clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(entity->getGuid(), componentId, componentType);
                        }

                        ImGui::EndPopup();
                    }
                }
            }
        }
      
        ImGui::PopID();
    }

    static std::vector<std::string> components = { "Rigidbody",    "Camera",
                                           "MeshRenderer",   "LineRenderer", "SpriteRenderer", "Light",
                                           "SphereCollider", "BoxCollider",  "Terrain" };
    size_t index;
    if (ImGui::BeginDropdownWindow("Add component", components, &index))
    {
        Component* component = nullptr;
        switch (index)
        {
        case 0:
            component = entity->addComponent<Rigidbody>();
            break;
        case 1:
            component = entity->addComponent<Camera>();
            break;
        case 2:
            component = entity->addComponent<MeshRenderer>();
            break;
        case 3:
            component = entity->addComponent<LineRenderer>();
            break;
        case 4:
            component = entity->addComponent<SpriteRenderer>();
            break;
        case 5:
            component = entity->addComponent<Light>();
            break;
        case 6:
            component = entity->addComponent<SphereCollider>();
            break;
        case 7:
            component = entity->addComponent<BoxCollider>();
            break;
        case 8:
            component = entity->addComponent<Terrain>();
            break;
        }
        
        if (component != nullptr) {
            clipboard.mSceneDirty = true;
        }

        ImGui::EndDropdownWindow();
    }
}