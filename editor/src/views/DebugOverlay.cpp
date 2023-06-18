#include "../../include/views/DebugOverlay.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "core/Material.h"
#include "core/Shader.h"
#include "core/Scene.h"

using namespace PhysicsEditor;

DebugOverlay::DebugOverlay()
{
    mMaxFPS = 0.0f;
}

DebugOverlay::~DebugOverlay()
{
}

void DebugOverlay::init(Clipboard& clipboard)
{
    mPerfQueue.setNumberOfSamples(100);
}

void DebugOverlay::update(Clipboard& clipboard)
{
    static ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoResize;

    static bool show_overlay = false;
    if (ImGui::IsKeyPressed(112, false))
    {
        show_overlay = !show_overlay;
    }

    ImGui::SetNextWindowBgAlpha(0.35f);

    if (show_overlay)
    {
        if (ImGui::Begin("Debug Overlay", NULL, overlay_flags))
        {
            if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
            {
                ImGui::SetWindowFocus("Debug Overlay");
            }

            ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
            if (ImGui::BeginTabBar("Debug Overlay Tab Bar", tab_bar_flags))
            {
                if (ImGui::BeginTabItem("Scene"))
                {
                    sceneTab(clipboard);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Shaders"))
                {
                    shaderTab(clipboard);
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }

        ImGui::End();
    }
}

void DebugOverlay::sceneTab(Clipboard& clipboard)
{
    ImGui::BeginColumns("Column Layout", 2, ImGuiColumnsFlags_GrowParentContentsSize);
    ImGui::SetColumnOffset(0, 0.0f);
    ImGui::SetColumnOffset(1, 300.0f);

    ImGui::Text("Framerate: %f", ImGui::GetIO().Framerate);

    const char* views[] = {"Inspector", "SceneView", "Hierarchy", "ProjectView", "Console"};

    for (int i = 0; i < static_cast<int>(View::Count); i++)
    {
        ImGui::Text(views[i]);
        ImGui::Indent(16.0f);
        ImGui::Text("Is open? %d\n", clipboard.mOpen[i]);
        ImGui::Text("Is opened this frame? %d\n", clipboard.mOpenedThisFrame[i]);
        ImGui::Text("Is closed this frame? %d\n", clipboard.mClosedThisFrame[i]);
        ImGui::Text("Is hovered? %d\n", clipboard.mHovered[i]);
        ImGui::Text("Is hovered this frame? %d\n", clipboard.mHoveredThisFrame[i]);
        ImGui::Text("Is unhovered this frame? %d\n", clipboard.mUnhoveredThisFrame[i]);
        ImGui::Text("Is focused? %d\n", clipboard.mFocused[i]);
        ImGui::Text("Is focused this frame? %d\n", clipboard.mFocusedThisFrame[i]);
        ImGui::Text("Is unfocused this frame? %d\n", clipboard.mUnfocusedThisFrame[i]);
        ImGui::Unindent(16.0f);
    }

    ImGui::NextColumn();
    ImGui::Text("Active project name: %s\n", clipboard.getProjectName().c_str());
    ImGui::Text("Active project path: %s\n", clipboard.getProjectPath().string().c_str());
    ImGui::Text("Active scene name: %s\n", clipboard.getSceneName().c_str());
    ImGui::Text("Active scene path: %s\n", clipboard.getScenePath().string().c_str());
    ImGui::Text("Active scene id: %s\n", clipboard.getSceneId().toString().c_str());

    ImGui::Text("Selected id: %s\n", clipboard.getSelectedId().toString().c_str());
    ImGui::Text("Selected type: %s\n", std::to_string((int)clipboard.getSelectedType()).c_str());

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    ImGui::Text("Scene count in world: %d\n", clipboard.getWorld()->getNumberOfScenes());
    for (size_t i = 0; i < clipboard.getWorld()->getNumberOfScenes(); i++)
    {
        PhysicsEngine::Scene* scene = clipboard.getWorld()->getSceneByIndex(i);

        ImGui::Text("Scene name: %s guid: %s\n", scene->getName().c_str(), scene->getGuid().toString().c_str());
    }

    ImGui::Text("Entity count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfEntities());

    ImGui::Text("Components");
    ImGui::Indent(16.0f);
    ImGui::Text("Transform count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents <PhysicsEngine::Transform>());
    ImGui::Text("MeshRenderer count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::MeshRenderer>());
    ImGui::Text("Light count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::Light>());
    ImGui::Text("Camera count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::Camera>());
    ImGui::Text("Rigidbody count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::Rigidbody>());
    ImGui::Text("SphereCollider count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::SphereCollider>());
    ImGui::Text("BoxCollider count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::BoxCollider>());
    ImGui::Text("CapsuleCollider count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::CapsuleCollider>());
    ImGui::Text("MeshCollider count: %d\n", clipboard.getWorld()->getActiveScene()->getNumberOfComponents<PhysicsEngine::MeshCollider>());
    ImGui::Unindent(16.0f);

    ImGui::Text("Assets");
    ImGui::Indent(16.0f);
    ImGui::Text("Shader count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Shader>());
    ImGui::Text("Mesh count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Mesh>());
    ImGui::Text("Material count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>());
    ImGui::Text("Texture2D count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Texture2D>());
    ImGui::Text("Cubemap count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Cubemap>());
    ImGui::Text("Font count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Font>());
    ImGui::Text("Sprite count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Sprite>());
    ImGui::Unindent(16.0f);

    ImGui::SliderFloat("MaxFPS", &mMaxFPS, 30.0f, 120.0f, "%.0f", 1.0f);

    mPerfQueue.addSample(ImGui::GetIO().Framerate);

    std::vector<float> perfData = mPerfQueue.getData();
    ImGui::PlotHistogram("##PerfPlot", &perfData[0], (int)perfData.size(), 0, nullptr, 0, mMaxFPS, ImVec2(400, 50));

    //PlotHistogram(const char* label, const float* values, int values_count, int values_offset = 0, const char* overlay_text = NULL, float scale_min = FLT_MAX, float scale_max = FLT_MAX, ImVec2 graph_size = ImVec2(0, 0), int stride = sizeof(float));

    ImGui::Text("RenderSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::RenderSystem>());
    ImGui::Text("PhysicsSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::PhysicsSystem>());
    ImGui::Text("CleanUpSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::CleanUpSystem>());
    ImGui::Text("GizmoSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::GizmoSystem>());
    ImGui::Text("FreeLookCameraSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::FreeLookCameraSystem>());

    ImGui::EndColumns();
}

void DebugOverlay::shaderTab(Clipboard& clipboard)
{
    static int index = 0;

    PhysicsEngine::Shader* selected = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Shader>(index);

    PhysicsEngine::Guid currentShaderId = selected->getGuid();

    if (ImGui::BeginCombo("Shader", (selected == nullptr ? "" : selected->getName()).c_str(), ImGuiComboFlags_None))
    {
        for (int i = 0; i < clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Shader>(); i++)
        {
            PhysicsEngine::Shader* s = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Shader>(i);

            std::string label = s->getName() + "##" + s->getGuid().toString();

            bool is_selected = (currentShaderId == s->getGuid());
            if (ImGui::Selectable(label.c_str(), is_selected))
            {
                selected = s;
                currentShaderId = s->getGuid();
                index = i;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    std::vector<PhysicsEngine::ShaderUniform> uniforms = selected->getUniforms();

    for (size_t i = 0; i < uniforms.size(); i++)
    {
        ImGui::Text(("Uniform: " + std::to_string(i)).c_str());
        ImGui::Indent(16.0f);
        {
            ImGui::Text("Data: "); ImGui::SameLine(); ImGui::Text(uniforms[i].mData);
            ImGui::Text("Name: "); ImGui::SameLine(); ImGui::Text(uniforms[i].mName.c_str());
            ImGui::Text("Type: "); ImGui::SameLine(); ImGui::Text(std::to_string(static_cast<int>(uniforms[i].mType)).c_str());
            //ImGui::Text("CachedHandle: "); ImGui::SameLine(); ImGui::Text(std::to_string(*reinterpret_cast<unsigned int*>(uniforms[i].mTex)).c_str());
            ImGui::Text("Uniform id: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mUniformId).c_str());
        }
        ImGui::Unindent(16.0f);
    }
}