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
    //Renderer::getRenderer()->createFramebuffer(256, 256, &mFBO, &mColor, &mDepth);

    //Renderer::getRenderer()->bindFramebuffer(mFBO);
    //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
    //Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    //Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    //Renderer::getRenderer()->unbindFramebuffer();
    mFBO->bind();
    mFBO->setViewport(0, 0, 256, 256);
    mFBO->clearColor(Color::black);
    mFBO->clearDepth(1.0f);
    mFBO->unbind();

    std::string vertexShader = "#version 430 core\n"
                               "in vec3 position;\n"
                               "in vec2 texCoord;\n"
                               "out vec2 TexCoord;\n"
                               "void main()\n"
                               "{\n"
                               "	gl_Position = vec4(position, 1.0);\n"
                               "   TexCoord = texCoord;\n"
                               "}";

    std::string fragmentShaderR = "#version 430 core\n"
                                  "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(texture(texture0, TexCoord).r, 0, 0, 1);\n"
                                  "}";
    std::string fragmentShaderG = "#version 430 core\n"
                                  "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(0, texture(texture0, TexCoord).g, 0, 1);\n"
                                  "}";
    std::string fragmentShaderB = "#version 430 core\n"
                                  "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(0, 0, texture(texture0, TexCoord).b, 1);\n"
                                  "}";
    std::string fragmentShaderA = "#version 430 core\n"
                                  "uniform sampler2D texture0;\n"
                                  "in vec2 TexCoord;\n"
                                  "out vec4 FragColor;\n"
                                  "void main()\n"
                                  "{\n"
                                  "    FragColor = vec4(texture(texture0, TexCoord).a,\n"
                                  "                     texture(texture0, TexCoord).a,\n"
                                  "                     texture(texture0, TexCoord).a, 1);\n"
                                  "}";

    /*ShaderStatus status;
    Renderer::getRenderer()->compile("Texture2DDrawer0", vertexShader, fragmentShaderR, "", &mProgramR, status);
    Renderer::getRenderer()->compile("Texture2DDrawer1", vertexShader, fragmentShaderG, "", &mProgramG, status);
    Renderer::getRenderer()->compile("Texture2DDrawer2", vertexShader, fragmentShaderB, "", &mProgramB, status);
    Renderer::getRenderer()->compile("Texture2DDrawer3", vertexShader, fragmentShaderA, "", &mProgramA, status);*/
    
    /*mTexLocR = Renderer::getRenderer()->findUniformLocation("texture0", mProgramR);
    mTexLocG = Renderer::getRenderer()->findUniformLocation("texture0", mProgramG);
    mTexLocB = Renderer::getRenderer()->findUniformLocation("texture0", mProgramB);
    mTexLocA = Renderer::getRenderer()->findUniformLocation("texture0", mProgramA);*/
    
    mProgramR = ShaderProgram::create();
    mProgramG = ShaderProgram::create();
    mProgramB = ShaderProgram::create();
    mProgramA = ShaderProgram::create();

    mProgramR->load("R",vertexShader, fragmentShaderR);
    mProgramG->load("G",vertexShader, fragmentShaderG);
    mProgramB->load("B",vertexShader, fragmentShaderB);
    mProgramA->load("A",vertexShader, fragmentShaderA);

    mProgramR->compile();
    mProgramG->compile();
    mProgramB->compile();
    mProgramA->compile();

    mTexLocR = mProgramR->findUniformLocation("texture0");
    mTexLocG = mProgramG->findUniformLocation("texture0");
    mTexLocB = mProgramB->findUniformLocation("texture0");
    mTexLocA = mProgramA->findUniformLocation("texture0");

    mCurrentTexId = Guid::INVALID;
    mDrawTex = -1;
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

    ImGui::Separator();

    if (mCurrentTexId != texture->getGuid())
    {
        mCurrentTexId = texture->getGuid();
        mDrawTex = *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle());
    }

    const std::array<const char *, 2> wrapMode = {"Repeat", "Clamp To Edge"};
    const std::array<const char *, 3> filterMode = {"Nearest", "Linear", "Bilinear"};

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
                mDrawTex = *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle());
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
            if (ImGui::Button("R"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramR->bind();
                mProgramR->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramR);
                //Renderer::getRenderer()->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramR->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramG->bind();
                mProgramG->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramG);
                //Renderer::getRenderer()->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
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
                mDrawTex = *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle());
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
            if (ImGui::Button("R"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramR->bind();
                mProgramR->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramR);
                //Renderer::getRenderer()->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramR->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramG->bind();
                mProgramG->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramG);
                //Renderer::getRenderer()->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramG->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
            if (ImGui::Button("B"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramB->bind();
                mProgramB->setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramB);
                //Renderer::getRenderer()->setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
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
                mDrawTex = *reinterpret_cast<unsigned int*>(texture->getNativeGraphics()->getHandle());
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
            if (ImGui::Button("R"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramR->bind();
                mProgramR->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramR);
                //Renderer::getRenderer()->setTexture2D(mTexLocR, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramR->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFF00FF00);
            if (ImGui::Button("G"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramG->bind();
                mProgramG->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramG);
                //Renderer::getRenderer()->setTexture2D(mTexLocG, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramG->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFF0000);
            if (ImGui::Button("B"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramB->bind();
                mProgramB->setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramB);
                //Renderer::getRenderer()->setTexture2D(mTexLocB, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramB->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, 0xFFFFFFFF);
            if (ImGui::Button("A"))
            {
                //mDrawTex = mColor;
                //Renderer::getRenderer()->bindFramebuffer(mFBO);
                //Renderer::getRenderer()->setViewport(0, 0, 256, 256);
                mDrawTex = *reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle());
                mFBO->bind();
                mFBO->setViewport(0, 0, 256, 256);
                mProgramA->bind();
                mProgramA->setTexture2D(mTexLocA, 0, texture->getNativeGraphics());
                //Renderer::getRenderer()->use(mProgramA);
                //Renderer::getRenderer()->setTexture2D(mTexLocA, 0, texture->getNativeGraphics());
                Renderer::getRenderer()->renderScreenQuad(mVAO);
                //Renderer::getRenderer()->unuse();
                //Renderer::getRenderer()->unbindFramebuffer();
                mProgramA->unbind();
                mFBO->unbind();
            }
            ImGui::PopStyleColor();
        }

        ImGui::Image((void *)(intptr_t)mDrawTex,
                     ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                     ImVec2(0, 0));

        ImGui::EndChild();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}