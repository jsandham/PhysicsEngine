#include "../../include/views/Inspector.h"

#include "imgui.h"
#include "../../include/imgui/imgui_extensions.h"

#include "../include/components/Light.h"
#include "../include/components/MeshRenderer.h"
#include "../include/components/ComponentTypes.h"

using namespace PhysicsEditor;

Inspector::Inspector() : mOpen(true)
{

}

Inspector::~Inspector()
{
}

void Inspector::init(Clipboard& clipboard)
{
}

void Inspector::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		mOpen = true;
	}

	if (!mOpen)
	{
		return;
	}

	if (ImGui::Begin("Inspector", &mOpen))
	{
		if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
		{
			ImGui::SetWindowFocus("Inspector");
		}
	}

	/*clipboard.mOpen[static_cast<int>(View::Inspector)] = isOpen();
	clipboard.mHovered[static_cast<int>(View::Inspector)] = isHovered();
	clipboard.mFocused[static_cast<int>(View::Inspector)] = isFocused();
	clipboard.mOpenedThisFrame[static_cast<int>(View::Inspector)] = openedThisFrame();
	clipboard.mHoveredThisFrame[static_cast<int>(View::Inspector)] = hoveredThisFrame();
	clipboard.mFocusedThisFrame[static_cast<int>(View::Inspector)] = focusedThisFrame();
	clipboard.mClosedThisFrame[static_cast<int>(View::Inspector)] = closedThisFrame();
	clipboard.mUnfocusedThisFrame[static_cast<int>(View::Inspector)] = unfocusedThisFrame();
	clipboard.mUnhoveredThisFrame[static_cast<int>(View::Inspector)] = unhoveredThisFrame();*/

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
	case InteractionType::RenderTexture:
		mRenderTextureDrawer.render(clipboard, clipboard.getSelectedId());
		break;
	}

	// draw selected entity
	if (clipboard.getSelectedType() == InteractionType::Entity)
	{
		drawEntity(clipboard);
	}

	ImGui::End();
}

void Inspector::drawEntity(Clipboard& clipboard)
{
	PhysicsEngine::Entity* entity = clipboard.getWorld()->getActiveScene()->getEntityByGuid(clipboard.getSelectedId());

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

	std::string name = entity->mName;

	std::vector<char> inputBuffer(256, '\0');
	std::copy(name.begin(), name.end(), inputBuffer.begin());

	ImGuiInputTextFlags options = ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue;
	if (ImGui::InputText("##Entity Header", &inputBuffer[0], inputBuffer.size(), options))
	{
		entity->mName = std::string(inputBuffer.begin(), inputBuffer.end());
	}

	ImGui::Text(("EntityId: " + entity->getGuid().toString()).c_str());

	ImGui::Separator();

	std::vector<std::pair<PhysicsEngine::Guid, int>> componentsOnEntity = entity->getComponentsOnEntity();
	for (size_t i = 0; i < componentsOnEntity.size(); i++)
	{
		PhysicsEngine::Guid componentId = componentsOnEntity[i].first;
		int componentType = componentsOnEntity[i].second;

		//ImGui::PushID(componentId.c_str());

		switch (componentType)
		{
		case PhysicsEngine::ComponentType<PhysicsEngine::Transform>::type: { mTransformDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::Rigidbody>::type: { mRigidbodyDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::Camera>::type: { mCameraDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::MeshRenderer>::type: { mMeshRendererDrawer.render(clipboard, componentId); break; }
		//case PhysicsEngine::ComponentType<PhysicsEngine::LineRenderer>::type: { mLineRendererDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::Light>::type: { mLightDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::BoxCollider>::type: { mBoxColliderDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::SphereCollider>::type: { mSphereColliderDrawer.render(clipboard, componentId); break; }
		//case PhysicsEngine::ComponentType<PhysicsEngine::CapsuleCollider>::type: { mCapsuleColliderDrawer.render(clipboard, componentId); break; }
		//case PhysicsEngine::ComponentType<PhysicsEngine::MeshCollider>::type: { mMeshColliderDrawer.render(clipboard, componentId); break; }
		case PhysicsEngine::ComponentType<PhysicsEngine::Terrain>::type: { mTerrainDrawer.render(clipboard, componentId); break; }
		}

		//ImGui::PopID();
	}

	/*static std::vector<std::string> components = { "Rigidbody",    "Camera",
										   "MeshRenderer",   "LineRenderer", "Light",
										   "SphereCollider", "BoxCollider",  "Terrain" };*/
	static std::vector<std::string> components = { "Rigidbody",    "Camera",
										   "MeshRenderer",   "Light",
										   "SphereCollider", "BoxCollider",  "Terrain" };
	size_t index;
	if (ImGui::BeginDropdownWindow("Add component", components, &index))
	{
		//PhysicsEngine::Component* component = nullptr;
		void* component = nullptr;
		switch (index)
		{
		case 0:
			component = entity->addComponent<PhysicsEngine::Rigidbody>();
			break;
		case 1:
			component = entity->addComponent<PhysicsEngine::Camera>();
			break;
		case 2:
			component = entity->addComponent<PhysicsEngine::MeshRenderer>();
			break;
		/*case 3:
			component = entity->addComponent<PhysicsEngine::LineRenderer>();
			break;*/
		case 3:
			component = entity->addComponent<PhysicsEngine::Light>();
			break;
		case 4:
			component = entity->addComponent<PhysicsEngine::SphereCollider>();
			break;
		case 5:
			component = entity->addComponent<PhysicsEngine::BoxCollider>();
			break;
		case 6:
			component = entity->addComponent<PhysicsEngine::Terrain>();
			break;
		}

		if (component != nullptr) {
			clipboard.mSceneDirty = true;
		}

		ImGui::EndDropdownWindow();
	}
}