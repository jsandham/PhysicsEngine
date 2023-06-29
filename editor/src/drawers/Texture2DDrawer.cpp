#include "../../include/drawers/Texture2DDrawer.h"
#include "../../include/EditorClipboard.h"

#include "core/Texture2D.h"
#include "graphics/RenderContext.h"

#include "imgui.h"

#include <array>

using namespace PhysicsEditor;

Texture2DDrawer::Texture2DDrawer()
{
    mScreenQuad = PhysicsEngine::RendererMeshes::getScreenQuad();
    mFBO = PhysicsEngine::Framebuffer::create(256, 256);
    
    mFBO->bind();
    mFBO->setViewport(0, 0, 256, 256);
    mFBO->clearColor(PhysicsEngine::Color::black);
    mFBO->clearDepth(1.0f);
    mFBO->unbind();

    mCurrentTexId = PhysicsEngine::Guid::INVALID;
    mDrawTex = nullptr;
}

Texture2DDrawer::~Texture2DDrawer()
{
    delete mFBO;
}

void Texture2DDrawer::render(Clipboard &clipboard, const PhysicsEngine::Guid& id)
{
    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    PhysicsEngine::Texture2D *texture = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Texture2D>(id);

    if (texture != nullptr)
    {
        ImGui::Separator();

        if (mCurrentTexId != texture->getGuid())
        {
            mCurrentTexId = texture->getGuid();
            mDrawTex = texture->getNativeGraphics()->getIMGUITexture();
        }

        const std::array<const char*, 5> wrapModes = { "Repeat", "Clamp To Edge", "Clamp to border", "Mirror repeat", "Mirror clamp to edge"};
        const std::array<const char*, 3> filterModes = { "Nearest", "Linear", "Bilinear" };

        static int activeWrapModeIndex = static_cast<int>(texture->getWrapMode());
        static int activeFilterModeIndex = static_cast<int>(texture->getFilterMode());

        // select wrap mode for texture
        if (ImGui::BeginCombo("Wrap Mode", wrapModes[activeWrapModeIndex]))
        {
            for (int n = 0; n < wrapModes.size(); n++)
            {
                bool is_selected = (wrapModes[activeWrapModeIndex] == wrapModes[n]);
                if (ImGui::Selectable(wrapModes[n], is_selected))
                {
                    activeWrapModeIndex = n;

                    texture->setWrapMode(static_cast<PhysicsEngine::TextureWrapMode>(n));
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
        if (ImGui::BeginCombo("Filter Mode", filterModes[activeFilterModeIndex]))
        {
            for (int n = 0; n < filterModes.size(); n++)
            {
                bool is_selected = (filterModes[activeFilterModeIndex] == filterModes[n]);
                if (ImGui::Selectable(filterModes[n], is_selected))
                {
                    activeFilterModeIndex = n;

                    texture->setFilterMode(static_cast<PhysicsEngine::TextureFilterMode>(n));
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
            PhysicsEngine::Shader* shaderR = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(PhysicsEngine::Guid("f1b412ca-6641-425c-b996-72ac78e9c709"));
            PhysicsEngine::Shader* shaderG = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(PhysicsEngine::Guid("1bec6e45-1cfb-4bb8-8cd9-9bb331e0459c"));
            PhysicsEngine::Shader* shaderB = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(PhysicsEngine::Guid("3d0fbdd9-bfbb-4add-9b1b-93eb79162f48"));
            PhysicsEngine::Shader* shaderA = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(PhysicsEngine::Guid("0a125454-09bd-4cad-bf80-b8c98ad72681"));

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

            if (texture->getFormat() == PhysicsEngine::TextureFormat::Depth)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                ImGui::Button("Depth");
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == PhysicsEngine::TextureFormat::RG)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RG"))
                {
                    mDrawTex = texture->getNativeGraphics()->getIMGUITexture();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == PhysicsEngine::TextureFormat::RGB)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RGB"))
                {
                    mDrawTex = texture->getNativeGraphics()->getIMGUITexture();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
                if (ImGui::Button("B"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramB->bind();
                    mProgramB->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramB->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }
            else if (texture->getFormat() == PhysicsEngine::TextureFormat::RGBA)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF000000);
                if (ImGui::Button("RGBA"))
                {
                    mDrawTex = texture->getNativeGraphics()->getIMGUITexture();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                if (ImGui::Button("R"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramR->bind();
                    mProgramR->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramR->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
                if (ImGui::Button("G"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramG->bind();
                    mProgramG->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramG->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
                if (ImGui::Button("B"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramB->bind();
                    mProgramB->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramB->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFFFFFF);
                if (ImGui::Button("A"))
                {
                    mDrawTex = mFBO->getColorTex()->getIMGUITexture();
                    mFBO->bind();
                    mFBO->setViewport(0, 0, 256, 256);
                    mProgramA->bind();
                    mProgramA->setTexture2D("texture0", 0, texture->getNativeGraphics()->getTexture());
                    mScreenQuad->draw();
                    mProgramA->unbind();
                    mFBO->unbind();
                }
                ImGui::PopStyleColor();
            }

            if (mDrawTex != nullptr)
            {
                if (PhysicsEngine::RenderContext::getRenderAPI() == PhysicsEngine::RenderAPI::OpenGL)
                {
                    // opengl
                    ImGui::Image((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(mDrawTex)),
                        ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                        ImVec2(0, 0));
                }
                else
                {
                    // directx
                    ImGui::Image(mDrawTex,
                        ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                        ImVec2(0, 0));
                }
            }

            ImGui::EndChild();
        }
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}

bool Texture2DDrawer::isHovered() const
{
    ImVec2 cursorPos = ImGui::GetMousePos();

    glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
    glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

    PhysicsEngine::Rect rect(min, max);

    return rect.contains(cursorPos.x, cursorPos.y);
}