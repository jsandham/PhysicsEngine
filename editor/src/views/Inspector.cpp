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
    else if (clipboard.getSelectedType() == InteractionType::RenderTexture)
    {
        mRenderTextureDrawer.render(clipboard, clipboard.getSelectedId());
    }

    // draw selected entity
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        drawEntity(clipboard);
    }
}

void Inspector::drawEntity(Clipboard &clipboard)
{
    Entity *entity = clipboard.getWorld()->getEntityById(clipboard.getSelectedId());

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

    ImGui::Text(("EntityId: " + entity->getId().toString()).c_str());

    ImGui::Separator();

    std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity();
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        Guid componentId = componentsOnEntity[i].first;
        int componentType = componentsOnEntity[i].second;

        ImGui::PushID(componentId.toString().c_str());

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
            else if (componentType == ComponentType<Terrain>::type)
            {
                drawer = &mTerrainDrawer;
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
                            clipboard.getWorld()->immediateDestroyComponent(entity->getId(), componentId, componentType);
                        }

                        ImGui::EndPopup();
                    }
                }
            }
        }
      
        ImGui::PopID();
    }

    std::string componentToAdd = "";
    std::vector<std::string> components = {"Rigidbody",    "Camera",
                                           "MeshRenderer",   "LineRenderer", "SpriteRenderer", "Light",
                                           "SphereCollider", "BoxCollider",  "Terrain"};

    if (ImGui::BeginDropdownWindow("Add component", components, componentToAdd))
    {
        Component* component = nullptr;
        if (componentToAdd == "Rigidbody")
        {
            component = entity->addComponent<Rigidbody>();
        }
        else if (componentToAdd == "Camera")
        {
            component = entity->addComponent<Camera>();
        }
        else if (componentToAdd == "MeshRenderer")
        {
            component = entity->addComponent<MeshRenderer>();
        }
        else if (componentToAdd == "LineRenderer")
        {
            component = entity->addComponent<LineRenderer>();
        }
        else if (componentToAdd == "SpriteRenderer")
        {
            component = entity->addComponent<SpriteRenderer>();
        }
        else if (componentToAdd == "Light")
        {
            component = entity->addComponent<Light>();
        }
        else if (componentToAdd == "SphereCollider")
        {
            component = entity->addComponent<SphereCollider>();
        }
        else if (componentToAdd == "BoxCollider")
        {
            component = entity->addComponent<BoxCollider>();
        }
        else if (componentToAdd == "Terrain")
        {
            component = entity->addComponent<Terrain>();
        }

        if (component != nullptr) {
            clipboard.mSceneDirty = true;
        }

        ImGui::EndDropdownWindow();
    }
}