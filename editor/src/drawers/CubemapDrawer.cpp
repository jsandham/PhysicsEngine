#include "../../include/drawers/CubemapDrawer.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/Cubemap.h"

#include "imgui.h"

using namespace PhysicsEditor;

CubemapDrawer::CubemapDrawer()
{
}

CubemapDrawer::~CubemapDrawer()
{
}

void CubemapDrawer::render(Clipboard &clipboard, const Guid& id)
{
	InspectorDrawer::render(clipboard, id);

	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

    Cubemap* cubemap = clipboard.getWorld()->getAssetByGuid<Cubemap>(id);

    if (cubemap != nullptr)
    {
        ImGui::Text(("Cubemap id: " + cubemap->getGuid().toString()).c_str());
        ImGui::Text(("Left tex id: " + cubemap->getTexId(CubemapFace::NegativeX).toString()).c_str());
        ImGui::Text(("Right tex id: " + cubemap->getTexId(CubemapFace::PositiveX).toString()).c_str());
        ImGui::Text(("Bottom tex id: " + cubemap->getTexId(CubemapFace::NegativeY).toString()).c_str());
        ImGui::Text(("Top tex id: " + cubemap->getTexId(CubemapFace::PositiveY).toString()).c_str());
        ImGui::Text(("Back tex id: " + cubemap->getTexId(CubemapFace::NegativeZ).toString()).c_str());
        ImGui::Text(("Front tex id: " + cubemap->getTexId(CubemapFace::PositiveZ).toString()).c_str());

        ImGui::Separator();

        Texture2D* leftTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::NegativeX));
        Texture2D* rightTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::PositiveX));
        Texture2D* bottomTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::NegativeY));
        Texture2D* topTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::PositiveY));
        Texture2D* backTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::NegativeZ));
        Texture2D* frontTex = clipboard.getWorld()->getAssetByGuid<Texture2D>(cubemap->getTexId(CubemapFace::PositiveZ));

        this->drawCubemapFaceTexture(clipboard, CubemapFace::NegativeX, cubemap, leftTex); ImGui::SameLine();
        this->drawCubemapFaceTexture(clipboard, CubemapFace::PositiveX, cubemap, rightTex);
        this->drawCubemapFaceTexture(clipboard, CubemapFace::NegativeY, cubemap, bottomTex); ImGui::SameLine();
        this->drawCubemapFaceTexture(clipboard, CubemapFace::PositiveY, cubemap, topTex);
        this->drawCubemapFaceTexture(clipboard, CubemapFace::NegativeZ, cubemap, backTex); ImGui::SameLine();
        this->drawCubemapFaceTexture(clipboard, CubemapFace::PositiveZ, cubemap, frontTex);
    }

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}

void CubemapDrawer::drawCubemapFaceTexture(Clipboard& clipboard, CubemapFace face, Cubemap* cubemap, Texture2D* texture)
{
    if (ImGui::ImageButton((void*)(intptr_t)(texture == nullptr ? 0 : *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle())),
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
        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("TEXTURE2D");
        if (payload != nullptr)
        {
            const PhysicsEngine::Guid* data = static_cast<const PhysicsEngine::Guid*>(payload->Data);

            cubemap->setTexId(face, *data);
            //material->onTextureChanged();
            clipboard.mModifiedAssets.insert(cubemap->getGuid());
        }
        ImGui::EndDragDropTarget();
    }
}