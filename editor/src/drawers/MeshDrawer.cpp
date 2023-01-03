#include "../../include/drawers/MeshDrawer.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/Mesh.h"

#include "imgui.h"

using namespace PhysicsEditor;

MeshDrawer::MeshDrawer()
{
    mActiveDrawModeIndex = 0;
    mWireframeOn = false;
    mResetModelMatrix = false;

    mFBO = Framebuffer::create(1000, 1000);

    mCameraUniform = RendererUniforms::getCameraUniform();

    mModel = glm::mat4(1.0f);
}

MeshDrawer::~MeshDrawer()
{
    delete mFBO;
}

void MeshDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Mesh *mesh = clipboard.getWorld()->getAssetByGuid<Mesh>(id);

    const int count = 4;
    const char *drawMode[] = {"Color", "Normals", "Tangents", "Binormal"};

    /*const Guid shaders[] = { clipboard.getWorld()->getAssetGuid("data\\shaders\\colorLit.shader"), 
                             clipboard.getWorld()->getAssetGuid("data\\shaders\\normal.shader"),
                             clipboard.getWorld()->getAssetGuid("data\\shaders\\tangent.shader"),
                             clipboard.getWorld()->getAssetGuid("data\\shaders\\binormal.shader") };*/
    const Guid shaders[] = { Guid("9cc784fd-1c70-4a2c-bf22-dbd18fdb39cb"), //colorLit
                             Guid("2437b7b7-11e8-4fc5-9e65-c0d3227de100"), //normal
                             Guid("6e11628b-f727-4b30-bf40-33834060dee1"), //tangent
                             Guid("183e29ba-4db4-4dbf-aed4-a1add5697dd9") }; //binormal

    // select draw mode for mesh
    if (ImGui::BeginCombo("##DrawMode", drawMode[mActiveDrawModeIndex]))
    {
        for (int n = 0; n < count; n++)
        {
            bool is_selected = (drawMode[mActiveDrawModeIndex] == drawMode[n]);
            if (ImGui::Selectable(drawMode[n], is_selected))
            {
                mActiveDrawModeIndex = n;

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
        mResetModelMatrix = true;
    }
    ImGui::SameLine();

    if (ImGui::Checkbox("Wireframe", &mWireframeOn))
    {
    }

    ImGui::Separator();

    if (mesh == nullptr)
    {
        return;
    }

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

    Shader *shader = clipboard.getWorld()->getAssetByGuid<Shader>(shaders[mActiveDrawModeIndex]);

    float meshRadius = mesh->getBounds().mRadius;

    mCameraUniform->setCameraPos(glm::vec3(0.0f, 0.0f, -4 * meshRadius));
    mCameraUniform->setView(glm::lookAt(glm::vec3(0.0f, 0.0f, -4 * meshRadius), glm::vec3(0.0f, 0.0f, -4 * meshRadius) + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0, 1.0f, 0.0f)));
    mCameraUniform->setProjection(glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 8 * meshRadius));
    mCameraUniform->copyToUniformsToDevice();
  
    shader->bind(static_cast<int64_t>(ShaderMacro::None));
    shader->setMat4("model", mModel);

    if (mActiveDrawModeIndex == 0)
    {
        shader->setVec3("lightDirection", glm::vec3(-1.0f, -1.0f, -1.0f));
        shader->setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
    }

    mFBO->bind();
    mFBO->setViewport(0, 0, 1000, 1000);
    mFBO->clearColor(Color(0.15f, 0.15f, 0.15f, 1.0f));
    mFBO->clearDepth(1.0f);

    shader->setInt("wireframe", 1);

    Renderer::getRenderer()->render(0, (int)mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO());

    if (mWireframeOn)
    {
        shader->setInt("wireframe", 0);

        Renderer::getRenderer()->render(0, (int)mesh->getVertices().size() / 3, mesh->getNativeGraphicsVAO(), true);
    }

    mFBO->unbind();

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
            mMouseX = nx;
            mMouseY = ny;
        }

        if (ImGui::IsWindowHovered() && io.MouseDown[0])
        {
            float diffX = mMouseX - nx;
            float diffY = mMouseY - ny;

            mModel = glm::rotate(mModel, 2 * diffX, glm::vec3(0, 1, 0));
            mModel = glm::rotate(mModel, 2 * diffY, glm::vec3(1, 0, 0));

            mMouseX = nx;
            mMouseY = ny;
        }

        // ImGui::GetForegroundDrawList()->AddRect(contentMin, contentMax, 0xFFFF0000);

        ImGui::Image((void *)(intptr_t)(*reinterpret_cast<unsigned int*>(mFBO->getColorTex()->getHandle())),
                     ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1),
                     ImVec2(0, 0));
    }

    if (mResetModelMatrix)
    {
        mModel = glm::mat4(1.0f);
        mResetModelMatrix = false;
    }

    ImGui::EndChild();

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}