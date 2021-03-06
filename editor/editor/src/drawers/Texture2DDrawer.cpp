#include "../../include/drawers/Texture2DDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorClipboard.h"
#include "../../include/EditorCommands.h"

#include "core/Texture2D.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include <array>

using namespace PhysicsEditor;

Texture2DDrawer::Texture2DDrawer()
{
    Graphics::createScreenQuad(&mVAO, &mVBO);
    Graphics::createFramebuffer(256, 256, &mFBO, &mColor, &mDepth);

    Graphics::bindFramebuffer(mFBO);
    Graphics::setViewport(0, 0, 256, 256);
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();

    std::string vertexShader = "in vec3 position;\n"
                               "in vec2 texCoord;\n"
                               "out vec2 TexCoord;\n"
                               "void main()\n"
                               "{\n"
                               "	gl_Position = vec4(position, 1.0);\n"
                               "   TexCoord = texCoord;\n"
                               "}";

    std::string fragmentShaderR = "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(texture(texture0, TexCoord).r, 0, 0, 1);\n"
                                  "}";
    std::string fragmentShaderG = "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(0, texture(texture0, TexCoord).g, 0, 1);\n"
                                  "}";
    std::string fragmentShaderB = "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(0, 0, texture(texture0, TexCoord).b, 1);\n"
                                  "}";
    std::string fragmentShaderA = "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(texture(texture0, TexCoord).a,\n"
                                  "                     texture(texture0, TexCoord).a,\n"
                                  "                     texture(texture0, TexCoord).a, 1);\n"
                                  "}";

    Graphics::compile(vertexShader, fragmentShaderR, "", &mProgramR);
    Graphics::compile(vertexShader, fragmentShaderG, "", &mProgramG);
    Graphics::compile(vertexShader, fragmentShaderB, "", &mProgramB);
    Graphics::compile(vertexShader, fragmentShaderA, "", &mProgramA);

    mTexLocR = Graphics::findUniformLocation("texture0", mProgramR);
    mTexLocG = Graphics::findUniformLocation("texture0", mProgramG);
    mTexLocB = Graphics::findUniformLocation("texture0", mProgramB);
    mTexLocA = Graphics::findUniformLocation("texture0", mProgramA);

    mCurrentTexId = Guid::INVALID;
    mDrawTex = -1;
}

Texture2DDrawer::~Texture2DDrawer()
{
}

void Texture2DDrawer::render(EditorClipboard &clipboard, Guid id)
{
    Texture2D *texture = clipboard.getWorld()->getAssetById<Texture2D>(id);

    ImGui::Separator();

    if (mCurrentTexId != texture->getId())
    {
        mCurrentTexId = texture->getId();
        mDrawTex = texture->getNativeGraphics();
    }

    const std::array<const char *, 2> wrapMode = {"Repeat", "Clamp To Edge"};
    const std::array<const char *, 3> filterMode = {"Nearest", "Linear", "Bilinear"};

    static int activeWrapModeIndex = 0;
    static int activeFilterModeIndex = 0;

    // select wrap mode for texture
    if (ImGui::BeginCombo("Warp Mode", wrapMode[activeWrapModeIndex]))
    {
        for (int n = 0; n < wrapMode.size(); n++)
        {
            bool is_selected = (wrapMode[activeWrapModeIndex] == wrapMode[n]);
            if (ImGui::Selectable(wrapMode[n], is_selected))
            {
                activeWrapModeIndex = n;

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

                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
        }
        ImGui::EndCombo();
    }

    // select aniso filtering for texture
    static int aniso = 0;
    if (ImGui::SliderInt("Aniso", &aniso, 0, 16))
    {
    }

    // Draw texture child window
    {
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
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramR);
                Graphics::setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramG);
                Graphics::setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
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
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramR);
                Graphics::setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramG);
                Graphics::setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
            if (ImGui::Button("B"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramB);
                Graphics::setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
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
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramR);
                Graphics::setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramG);
                Graphics::setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
            if (ImGui::Button("B"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramB);
                Graphics::setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFFFFFF);
            if (ImGui::Button("A"))
            {
                mDrawTex = mColor;
                Graphics::bindFramebuffer(mFBO);
                Graphics::setViewport(0, 0, 256, 256);
                Graphics::use(mProgramA);
                Graphics::setTexture2D(mTexLocA, 0, texture->getNativeGraphics());
                Graphics::renderScreenQuad(mVAO);
                Graphics::unuse();
                Graphics::unbindFramebuffer();
            }
            ImGui::PopStyleColor();
        }

        ImGui::Image((void *)(intptr_t)mDrawTex,
                     ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                     ImVec2(0, 0));

        ImGui::EndChild();
    }
}