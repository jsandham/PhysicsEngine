#include <algorithm>

#include "../../include/views/Hierarchy.h"
#include "../../include/ProjectDatabase.h"
#include "../../include/imgui/imgui_extensions.h"
#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy() : mOpen(true)
{

}

Hierarchy::~Hierarchy()
{
}

void Hierarchy::init(Clipboard& clipboard)
{
}

void Hierarchy::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		mOpen = true;
	}

	if (!mOpen)
	{
		return;
	}

	if (ImGui::Begin("Hierarchy", &mOpen))
	{
		if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
		{
			ImGui::SetWindowFocus("Hierarchy");
		}
	}

	mWindowPos = ImGui::GetWindowPos();
	mContentMin = ImGui::GetWindowContentRegionMin();
	mContentMax = ImGui::GetWindowContentRegionMax();

	mContentMin.x += mWindowPos.x;
	mContentMin.y += mWindowPos.y;
	mContentMax.x += mWindowPos.x;
	mContentMax.y += mWindowPos.y;


	/*clipboard.mOpen[static_cast<int>(View::Hierarchy)] = isOpen();
	clipboard.mHovered[static_cast<int>(View::Hierarchy)] = isHovered();
	clipboard.mFocused[static_cast<int>(View::Hierarchy)] = isFocused();
	clipboard.mOpenedThisFrame[static_cast<int>(View::Hierarchy)] = openedThisFrame();
	clipboard.mHoveredThisFrame[static_cast<int>(View::Hierarchy)] = hoveredThisFrame();
	clipboard.mFocusedThisFrame[static_cast<int>(View::Hierarchy)] = focusedThisFrame();
	clipboard.mClosedThisFrame[static_cast<int>(View::Hierarchy)] = closedThisFrame();
	clipboard.mUnfocusedThisFrame[static_cast<int>(View::Hierarchy)] = unfocusedThisFrame();
	clipboard.mUnhoveredThisFrame[static_cast<int>(View::Hierarchy)] = unhoveredThisFrame();*/

	if (clipboard.mSceneOpened)
	{
		mEntries.resize(clipboard.getWorld()->getActiveScene()->getNumberOfEntities());

		size_t index = 0;
		for (size_t i = 0; i < clipboard.getWorld()->getActiveScene()->getNumberOfEntities(); i++)
		{
			PhysicsEngine::Entity* entity = clipboard.getWorld()->getActiveScene()->getEntityByIndex(i);
			if (entity->mHide == PhysicsEngine::HideFlag::None)
			{
				mEntries[index] = (int)i;
				index++;
			}
		}

		mEntries.resize(index);

		// Set selected entity in hierarchy
		int selectedIndex;
		if (clipboard.getSelectedType() == InteractionType::Entity)
		{
			selectedIndex = clipboard.getWorld()->getIndexOf(clipboard.getSelectedId());
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

		ImGuiListClipper clipper;
		clipper.Begin((int)mEntries.size());
		while (clipper.Step())
		{
			for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
			{
				PhysicsEngine::Entity* entity = clipboard.getWorld()->getActiveScene()->getEntityByIndex(mEntries[i]);
				static bool selected = false;
				char buf1[64];
				std::size_t len = std::min(size_t(64 - 1), entity->mName.length());
				strncpy(buf1, entity->mName.c_str(), len);
				buf1[len] = '\0';

				bool edited = false;
				/*if (ImGui::SelectableInput(entity->getGuid().c_str(), selectedIndex == mEntries[i], &edited,
					ImGuiSelectableFlags_DrawHoveredWhenHeld, buf1, IM_ARRAYSIZE(buf1)))*/
				if (ImGui::SelectableInput(entity->getGuid().c_str(), selectedIndex == mEntries[i], &edited,
					ImGuiSelectableFlags_None, buf1, IM_ARRAYSIZE(buf1)))
				{
					entity->mName = std::string(buf1);

					clipboard.setSelectedItem(InteractionType::Entity, entity->getGuid());
				}

				if (edited)
				{
					clipboard.mSceneDirty = true;
				}

				if (ImGui::BeginDragDropSource())
				{
					const void* data = static_cast<const void*>(entity->getGuid().c_str());

					ImGui::SetDragDropPayload("ENTITY_GUID", data, sizeof(PhysicsEngine::Guid));
					ImGui::Text(entity->mName.c_str());
					ImGui::EndDragDropSource();
				}
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
		ImRect rect(getContentMin(), getContentMax());
		if (ImGui::BeginDragDropTargetCustom(rect, ImGui::GetCurrentWindow()->ID))
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
	}

	ImGui::End();
}

ImVec2 Hierarchy::getWindowPos() const
{
	return mWindowPos;
}

ImVec2 Hierarchy::getContentMin() const
{
	return mContentMin;
}

ImVec2 Hierarchy::getContentMax() const
{
	return mContentMax;
}