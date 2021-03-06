#include "../../include/drawers/MeshDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/InternalShaders.h"
#include "core/Mesh.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

MeshDrawer::MeshDrawer()
{
    activeDrawModeIndex = 0;
    wireframeOn = false;
    resetModelMatrix = false;

    Graphics::createFramebuffer(1000, 1000, &mFBO, &mColor, &mDepth);

    Graphics::createGlobalCameraUniforms(cameraUniform);

    model = glm::mat4(1.0f);
}

MeshDrawer::~MeshDrawer()
{
}

void MeshDrawer::render(EditorClipboard &clipboard, Guid id)
{
    Mesh *mesh = clipboard.getWorld()->getAssetById<Mesh>(id);

    const int count = 3;
    const char *drawMode[] = {"Color", "Normals", "Tangents"};

    const Guid shaders[] = {clipboard.getWorld()->getColorLitShaderId(), clipboard.getWorld()->getNormalLitShaderId(),
                            clipboard.getWorld()->getTangentLitShaderId()};

    // select draw mode for mesh
    if (ImGui::BeginCombo("##DrawMode", drawMode[activeDrawModeIndex]))
    {
        for (int n = 0; n < count; n++)
        {
            bool is_selected = (drawMode[activeDrawModeIndex] == drawMode[n]);
            if (ImGui::Selectable(drawMode[n], is_selected))
            {
                activeDrawModeIndex = n;

                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();

    if (ImGui::Button("Reset"))
    {
        resetModelMatrix = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Wireframe", &wireframeOn))
    {
    }

    ImGui::Separator();

    ImGui::Text("Vertices");
    ImGui::Indent();
    ImGui::Text(("Positions: " + std::to_string(mesh->getVertices().size())).c_str());
    ImGui::Text(("Normals: " + std::to_string(mesh->getNormals().size())).c_str());
    ImGui::Text(("UV: " + std::to_string(mesh->getTexCoords().size())).c_str());
    ImGui::Unindent();

    ImGui::Text("Submesh count");
    ImGui::Indent();
    for (int i = 0; i < mesh->getSubMeshCount(); i++)
    {
        int startIndex = mesh->getSubMeshStartIndex(i);
        int endIndex = mesh->getSubMeshEndIndex(i);

        std::string str = std::to_string(i) + ". start index: " + std::to_string(startIndex) +
                          " end index: " + std::to_string(endIndex);
        ImGui::Text(str.c_str());
    }
    ImGui::Unindent();

    ImGui::Text("Bounds");
    ImGui::Indent();
    ImGui::Text(("Centre: " + std::to_string(mesh->getBounds().mCentre.x) + " " +
                 std::to_string(mesh->getBounds().mCentre.y) + " " + std::to_string(mesh->getBounds().mCentre.z))
                    .c_str());
    ImGui::Text(("Radius: " + std::to_string(mesh->getBounds().mRadius)).c_str());
    ImGui::Unindent();

    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    // Draw mesh preview child window
    ImGui::Text("Preview");

    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(shaders[activeDrawModeIndex]);
    int shaderProgram = shader->getProgramFromVariant(ShaderVariant::None);

    float meshRadius = mesh->getBounds().mRadius;

    cameraUniform.mCameraPos = glm::vec3(0.0f, 0.0f, -4 * meshRadius);
    cameraUniform.mView = glm::lookAt(cameraUniform.mCameraPos, cameraUniform.mCameraPos + glm::vec3(0.0f, 0.0f, 1.0f),
                                      glm::vec3(0.0, 1.0f, 0.0f));
    cameraUniform.mProjection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 8 * meshRadius);
    Graphics::setGlobalCameraUniforms(cameraUniform);

    shader->use(shaderProgram);
    shader->setMat4("model", model);
    shader->setVec3("lightDirection", glm::vec3(-1.0f, -1.0f, -1.0f));

    if (activeDrawModeIndex == 0)
    {
        shader->setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
    }

    Graphics::bindFramebuffer(mFBO);
    Graphics::setViewport(0, 0, 1000, 1000);
    Graphics::clearFrambufferColor(Color(0.0f, 0.0, 0.0, 1.0f));
    Graphics::clearFramebufferDepth(1.0f);

    shader->setInt("wireframe", 1);

    Graphics::render(0, mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO());

    if (wireframeOn)
    {
        shader->setInt("wireframe", 0);

        Graphics::render(0, mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO(), true);
    }

    Graphics::unbindFramebuffer();

    if (ImGui::BeginChild("MeshPreviewWindow",
                          ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true,
                          ImGuiWindowFlags_None))
    {
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
        ImVec2 contentMax = ImGui::GetWindowContentRegionMax();

        contentMin.x += windowPos.x;
        contentMin.y += windowPos.y;
        contentMax.x += windowPos.x;
        contentMax.y += windowPos.y;

        ImGuiIO &io = ImGui::GetIO();
        float contentWidth = (contentMax.x - contentMin.x);
        float contentHeight = (contentMax.y - contentMin.y);
        float mousePosX = std::min(std::max(io.MousePos.x - contentMin.x, 0.0f), contentWidth);
        float mousePosY = contentHeight - std::min(std::max(io.MousePos.y - contentMin.y, 0.0f), contentHeight);

        float nx = mousePosX / contentWidth;
        float ny = mousePosY / contentHeight;

        // Update selected entity
        if (ImGui::IsWindowHovered() && io.MouseClicked[0])
        {
            mouseX = nx;
            mouseY = ny;
        }

        if (ImGui::IsWindowHovered() && io.MouseDown[0])
        {
            float diffX = mouseX - nx;
            float diffY = mouseY - ny;

            model = glm::rotate(model, 2 * diffX, glm::vec3(0, 1, 0));
            model = glm::rotate(model, 2 * diffY, glm::vec3(1, 0, 0));

            mouseX = nx;
            mouseY = ny;
        }

        // ImGui::GetForegroundDrawList()->AddRect(contentMin, contentMax, 0xFFFF0000);

        ImGui::Image((void *)(intptr_t)mColor,
                     ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                     ImVec2(0, 0));
    }

    if (resetModelMatrix)
    {
        model = glm::mat4(1.0f);
        resetModelMatrix = false;
    }

    ImGui::EndChild();
}