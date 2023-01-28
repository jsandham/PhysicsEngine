#include "../../include/drawers/Texture2DDrawer.h"
#include "../../include/EditorClipboard.h"

#include "core/Texture2D.h"

#include "imgui.h"

#include <array>

using namespace PhysicsEditor;

Texture2DDrawer::Texture2DDrawer()
{
    Renderer::getRenderer()->createScreenQuad(&mVAO, &mVBO);
    mFBO = Framebuffer::create(256, 256);
    
    mFBO->bind();
    mFBO->setViewport(0, 0, 256, 256);
    mFBO->clearColor(Color::black);
    mFBO->clearDepth(1.0f);
    mFBO->unbind();

    mCurrentTexId = Guid::INVALID;
    mDrawTex = nullptr;
}

Texture2DDrawer::~Texture2DDrawer()
{
    delete mFBO;
}

void Texture2DDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Texture2D *texture = clipboard.getWorld()->getAssetByGuid<Texture2D>(id);

    if (texture != nullptr)
    {
        ImGui::Separator();

        if (mCurrentTexId != texture->getGuid())
        {
            mCurrentTexId = texture->getGuid();
            mDrawTex = texture->getNativeGraphics();
        }

        const std::array<const char*, 2> wrapMode = { "Repeat", "Clamp To Edge" };
        const std::array<const char*, 3> filterMode = { "Nearest", "Linear", "Bilinear" };

        static int activeWrapModeIndex = static_cast<int>(texture->getWrapMode());
        static int activeFilterModeIndex = static_cast<int>(texture->getFilterMode());

        // select wrap mode for texture
        if (ImGui::BeginCombo("Wrap Mode", wrapMode[activeWrapModeIndex]))
        {
            for (int n = 0; n < wrapMode.size(); n++)
            {
                bool is_selected = (wrapMode[activeWrapModeIndex] == wrapMode[n]);
                if (ImGui::Selectable(wrapMode[n], is_selected))
                {
                    activeWrapModeIndex = n;

                    texture->setWrapMode(static_cast<TextureWrapMode>(n));
                    clipboard.mModifiedAssets.insert(texture->getGuid());

                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            ImGui::EndCombo();
        }

        // select filter mode for texture
        if (ImGui::BeginCombo("Filter Mode", filterMode[activeFilterModeIndex]))
        {
            for (int n = 0; n < filterMode.size(); n++)
            {
                bool is_selected = (filterMode[activeFilterModeIndex] == filterMode[n]);
                if (ImGui::Selectable(filterMode[n], is_selected))
                {
                    activeFilterModeIndex = n;

                    texture->setFilterMode(static_cast<TextureFilterMode>(n));
                    clipboard.mModifiedAssets.insert(texture->getGuid());

                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            ImGui::EndCombo();
        }

        // select aniso filtering for texture
        static int aniso = texture->getAnisoLevel();
        if (ImGui::SliderInt("Aniso", &aniso, 1, 16))
        {
            texture->setAnisoLevel(aniso);
            clipboard.mModifiedAssets.insert(texture->getGuid());
        }

        // Draw texture child window
        {
            Shader* shaderR = clipboard.getWorld()->getAssetByGuid<Shader>(Guid("f1b412ca-6641-425c-b996-72ac78e9c709"));
            Shader* shaderG = clipboard.getWorld()->getAssetByGuid<Shader>(Guid("1bec6e45-1cfb-4bb8-8cd9-9bb331e0459c"));
            Shader* shaderB = clipboard.getWorld()->getAssetByGuid<Shader>(Guid("3d0fbdd9-bfbb-4add-9b1b-93eb79162f48"));
            Shader* shaderA = clipboard.getWorld()->getAssetByGuid<Shader>(Guid("0a125454-09bd-4cad-bf80-b8c98ad72681"));

            assert(shaderR != nullptr);
            assert(shaderG != nullptr);
            assert(shaderB != nullptr);
            assert(shaderA != nullptr);

            mProgramR = shaderR->getProgramFromVariant(0);
            mProgramG = shaderG->getProgramFromVariant(0);
            mProgramB = shaderB->getProgramFromVariant(0);
            mProgramA = shaderA->getProgramFromVariant(0);

            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar;
            ImGui::BeginChild("DrawTextureWindow",
                ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true,
                window_flags);

            if (texture->getFormat() == TextureFormat::Depth)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                ImGui::Button("Depth");
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == TextureFormat::RG)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RG"))
                {
                    mDrawTex = texture->getNativeGraphics();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == TextureFormat::RGB)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RGB"))
                {
                    mDrawTex = texture->getNativeGraphics();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
                if (ImGui::Button("B"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramB->bind();
                    mProgramB->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramB->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == TextureFormat::RGBA)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RGBA"))
                {
                    mDrawTex = texture->getNativeGraphics();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
                if (ImGui::Button("B"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramB->bind();
                    mProgramB->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramB->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFFFFFF);
                if (ImGui::Button("A"))
                {
                    mDrawTex = mFBO->getColorTex();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramA->bind();
                    mProgramA->setTexture2D("texture0", 0, texture->getNativeGraphics());
                    Renderer::getRenderer()->renderScreenQuad(mVAO);
                    mProgramA->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }

            ImGui::Image((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(mDrawTex->getHandle())),
                ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                ImVec2(0, 0));

            ImGui::EndChild();
        }
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}