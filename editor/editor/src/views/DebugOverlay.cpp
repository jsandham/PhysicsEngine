#include "../../include/views/DebugOverlay.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "core/Material.h"
#include "core/Shader.h"

using namespace PhysicsEditor;

DebugOverlay::DebugOverlay() : Window("Debug Overlay")
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
    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("Debug Overlay", tab_bar_flags))
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

void DebugOverlay::sceneTab(Clipboard& clipboard)
{
    ImGui::BeginColumns("Column Layout", 2, ImGuiColumnsFlags_GrowParentContentsSize);//false);
    ImGui::SetColumnOffset(0, 0.0f);
    ImGui::SetColumnOffset(1, 300.0f);

    ImGui::Text("Framerate: %f", ImGui::GetIO().Framerate);

    ImGui::Text("Is SceneView open? %d\n", clipboard.mSceneViewOpen);
    ImGui::Text("Is Inspector open? %d\n", clipboard.mInspectorOpen);
    ImGui::Text("Is Hierarchy open? %d\n", clipboard.mHierarchyOpen);
    ImGui::Text("Is Console open? %d\n", clipboard.mConsoleOpen);
    ImGui::Text("Is ProjectView open? %d\n", clipboard.mProjectViewOpen);

    ImGui::Text("Is SceneView opened this frame? %d\n", clipboard.mSceneViewOpenedThisFrame);
    ImGui::Text("Is Inspector opened this frame? %d\n", clipboard.mInspectorOpenedThisFrame);
    ImGui::Text("Is Hierarchy opened this frame? %d\n", clipboard.mHierarchyOpenedThisFrame);
    ImGui::Text("Is Console opened this frame? %d\n", clipboard.mConsoleOpenedThisFrame);
    ImGui::Text("Is ProjectView opened this frame? %d\n", clipboard.mProjectViewOpenedThisFrame);

    ImGui::Text("Is SceneView closed this frame? %d\n", clipboard.mSceneViewClosedThisFrame);
    ImGui::Text("Is Inspector closed this frame? %d\n", clipboard.mInspectorClosedThisFrame);
    ImGui::Text("Is Hierarchy closed this frame? %d\n", clipboard.mHierarchyClosedThisFrame);
    ImGui::Text("Is Console closed this frame? %d\n", clipboard.mConsoleClosedThisFrame);
    ImGui::Text("Is ProjectView closed this frame? %d\n", clipboard.mProjectViewClosedThisFrame);

    ImGui::Text("Is SceneView hovered? %d\n", clipboard.mSceneViewHovered);
    ImGui::Text("Is Inspector hovered? %d\n", clipboard.mInspectorHovered);
    ImGui::Text("Is Hierarchy hovered? %d\n", clipboard.mHierarchyHovered);
    ImGui::Text("Is Console hovered? %d\n", clipboard.mConsoleHovered);
    ImGui::Text("Is ProjectView hovered? %d\n", clipboard.mProjectViewHovered);

    ImGui::Text("Is SceneView hovered this frame? %d\n", clipboard.mSceneViewHoveredThisFrame);
    ImGui::Text("Is Inspector hovered this frame? %d\n", clipboard.mInspectorHoveredThisFrame);
    ImGui::Text("Is Hierarchy hovered this frame? %d\n", clipboard.mHierarchyHoveredThisFrame);
    ImGui::Text("Is Console hovered this frame? %d\n", clipboard.mConsoleHoveredThisFrame);
    ImGui::Text("Is ProjectView hovered this frame? %d\n", clipboard.mProjectViewHoveredThisFrame);

    ImGui::Text("Is SceneView unhovered this frame? %d\n", clipboard.mSceneViewUnhoveredThisFrame);
    ImGui::Text("Is Inspector unhovered this frame? %d\n", clipboard.mInspectorUnhoveredThisFrame);
    ImGui::Text("Is Hierarchy unhovered this frame? %d\n", clipboard.mHierarchyUnhoveredThisFrame);
    ImGui::Text("Is Console unhovered this frame? %d\n", clipboard.mConsoleUnhoveredThisFrame);
    ImGui::Text("Is ProjectView unhovered this frame? %d\n", clipboard.mProjectViewUnhoveredThisFrame);

    ImGui::Text("Is SceneView focused? %d\n", clipboard.mSceneViewFocused);
    ImGui::Text("Is Inspector focused? %d\n", clipboard.mInspectorFocused);
    ImGui::Text("Is Hierarchy focused? %d\n", clipboard.mHierarchyFocused);
    ImGui::Text("Is Console focused? %d\n", clipboard.mConsoleFocused);
    ImGui::Text("Is ProjectView focused? %d\n", clipboard.mProjectViewFocused);

    ImGui::Text("Is SceneView focused this frame? %d\n", clipboard.mSceneViewFocusedThisFrame);
    ImGui::Text("Is Inspector focused this frame? %d\n", clipboard.mInspectorFocusedThisFrame);
    ImGui::Text("Is Hierarchy focused this frame? %d\n", clipboard.mHierarchyFocusedThisFrame);
    ImGui::Text("Is Console focused this frame? %d\n", clipboard.mConsoleFocusedThisFrame);
    ImGui::Text("Is ProjectView focused this frame? %d\n", clipboard.mProjectViewFocusedThisFrame);

    ImGui::Text("Is SceneView unfocused this frame? %d\n", clipboard.mSceneViewUnfocusedThisFrame);
    ImGui::Text("Is Inspector unfocused this frame? %d\n", clipboard.mInspectorUnfocusedThisFrame);
    ImGui::Text("Is Hierarchy unfocused this frame? %d\n", clipboard.mHierarchyUnfocusedThisFrame);
    ImGui::Text("Is Console unfocused this frame? %d\n", clipboard.mConsoleUnfocusedThisFrame);
    ImGui::Text("Is ProjectView unfocused this frame? %d\n", clipboard.mProjectViewUnfocusedThisFrame);

    ImGui::NextColumn();
    ImGui::Text("Active project name: %s\n", clipboard.getProjectName().c_str());
    ImGui::Text("Active project path: %s\n", clipboard.getProjectPath().c_str());
    ImGui::Text("Active scene name: %s\n", clipboard.getSceneName().c_str());
    ImGui::Text("Active scene path: %s\n", clipboard.getScenePath().c_str());
    ImGui::Text("Active scene id: %s\n", clipboard.getSceneId().toString().c_str());

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    ImGui::Text("Scene count in world: %d\n", clipboard.getWorld()->getNumberOfScenes());
    ImGui::Text("Entity count in world: %d\n", clipboard.getWorld()->getNumberOfEntities());

    ImGui::Text("Components");
    ImGui::Indent(16.0f);
    ImGui::Text("Transform count: %d\n", clipboard.getWorld()->getNumberOfComponents <PhysicsEngine::Transform>());
    ImGui::Text("MeshRenderer count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::MeshRenderer>());
    ImGui::Text("Light count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::Light>());
    ImGui::Text("Camera count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::Camera>());
    ImGui::Text("Rigidbody count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::Rigidbody>());
    ImGui::Text("SphereCollider count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::SphereCollider>());
    ImGui::Text("BoxCollider count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::BoxCollider>());
    ImGui::Text("CapsuleCollider count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::CapsuleCollider>());
    ImGui::Text("MeshCollider count: %d\n", clipboard.getWorld()->getNumberOfComponents<PhysicsEngine::MeshCollider>());
    ImGui::Unindent(16.0f);

    ImGui::Text("Assets");
    ImGui::Indent(16.0f);
    ImGui::Text("Shader count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Shader>());
    ImGui::Text("Mesh count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Mesh>());
    ImGui::Text("Material count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>());
    ImGui::Text("Texture2D count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Texture2D>());
    ImGui::Text("Texture3D count: %d\n", clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Texture3D>());
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
    ImGui::Text("EditorCameraSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsEngine::EditorCameraSystem>());

    ImGui::EndColumns();
}

void DebugOverlay::shaderTab(Clipboard& clipboard)
{
    static int index = 0;

    PhysicsEngine::Shader* selected = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Shader>(index);

    PhysicsEngine::Guid currentShaderId = selected->getId();

    if (ImGui::BeginCombo("Shader", (selected == nullptr ? "" : selected->getName()).c_str(), ImGuiComboFlags_None))
    {
        for (int i = 0; i < clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Shader>(); i++)
        {
            PhysicsEngine::Shader* s = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Shader>(i);

            std::string label = s->getName() + "##" + s->getId().toString();

            bool is_selected = (currentShaderId == s->getId());
            if (ImGui::Selectable(label.c_str(), is_selected))
            {
                selected = s;
                currentShaderId = s->getId();
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
            //ImGui::Text("ShortName: "); ImGui::SameLine(); ImGui::Text(uniforms[i].mShortName);
            //ImGui::Text("BlockName: "); ImGui::SameLine(); ImGui::Text(uniforms[i].mBlockName);
            //ImGui::Text("NameLength: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mNameLength).c_str());
            //ImGui::Text("Size: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mSize).c_str());
            ImGui::Text("Type: "); ImGui::SameLine(); ImGui::Text(std::to_string(static_cast<int>(uniforms[i].mType)).c_str());
            //ImGui::Text("Variant: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mVariant).c_str());
            ImGui::Text("Location: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mLocation).c_str());
            //ImGui::Text("Index: "); ImGui::SameLine(); ImGui::Text(std::to_string(uniforms[i].mIndex).c_str());
        }
        ImGui::Unindent(16.0f);
    }
}