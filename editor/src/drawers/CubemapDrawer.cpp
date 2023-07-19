#include "../../include/drawers/CubemapDrawer.h"
#include "../../include/imgui/imgui_extensions.h"
#include "../../include/ProjectDatabase.h"

#include "core/Cubemap.h"

#include "imgui.h"

using namespace PhysicsEditor;

CubemapDrawer::CubemapDrawer()
{
}

CubemapDrawer::~CubemapDrawer()
{
}

void CubemapDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	PhysicsEngine::Cubemap* cubemap = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Cubemap>(id);

	if (cubemap != nullptr)
	{
		ImGui::Text(("Cubemap id: " + cubemap->getGuid().toString()).c_str());
		ImGui::Text(("Left tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeX).toString()).c_str());
		ImGui::Text(("Right tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveX).toString()).c_str());
		ImGui::Text(("Bottom tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeY).toString()).c_str());
		ImGui::Text(("Top tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveY).toString()).c_str());
		ImGui::Text(("Back tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeZ).toString()).c_str());
		ImGui::Text(("Front tex id: " + cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveZ).toString()).c_str());

		ImGui::Separator();

		PhysicsEngine::Texture2D* leftTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeX));
		PhysicsEngine::Texture2D* rightTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveX));
		PhysicsEngine::Texture2D* bottomTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeY));
		PhysicsEngine::Texture2D* topTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveY));
		PhysicsEngine::Texture2D* backTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::NegativeZ));
		PhysicsEngine::Texture2D* frontTex = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(cubemap->getTexId(PhysicsEngine::CubemapFace::PositiveZ));

		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::NegativeX, cubemap, leftTex); ImGui::SameLine();
		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::PositiveX, cubemap, rightTex);
		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::NegativeY, cubemap, bottomTex); ImGui::SameLine();
		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::PositiveY, cubemap, topTex);
		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::NegativeZ, cubemap, backTex); ImGui::SameLine();
		this->drawCubemapFaceTexture(clipboard, PhysicsEngine::CubemapFace::PositiveZ, cubemap, frontTex);
	}

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}

void CubemapDrawer::drawCubemapFaceTexture(Clipboard& clipboard, PhysicsEngine::CubemapFace face, PhysicsEngine::Cubemap* cubemap, PhysicsEngine::Texture2D* texture)
{
	if (ImGui::ImageButton((void*)(intptr_t)(texture == nullptr ? 0 : *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getIMGUITexture())),
		ImVec2(80, 80),
		ImVec2(1, 1),
		ImVec2(0, 0),
		0,
		ImVec4(1, 1, 1, 1),
		ImVec4(1, 1, 1, 0.5)))
	{
		if (texture != nullptr)
		{
			clipboard.setSelectedItem(InteractionType::Texture2D, texture->getGuid());
			clipboard.mModifiedAssets.insert(cubemap->getGuid());
		}
	}

	if (ImGui::BeginDragDropTarget())
	{
		const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("TEXTURE2D_PATH");
		if (payload != nullptr)
		{
			const char* data = static_cast<const char*>(payload->Data);

			cubemap->setTexId(face, ProjectDatabase::getGuid(data));
			//material->onTextureChanged();
			clipboard.mModifiedAssets.insert(cubemap->getGuid());
		}
		ImGui::EndDragDropTarget();
	}
}

bool CubemapDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}