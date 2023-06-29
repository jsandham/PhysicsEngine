#include "../../include/drawers/LineRendererDrawer.h"

#include "components/LineRenderer.h"

#include "imgui.h"

using namespace PhysicsEditor;

LineRendererDrawer::LineRendererDrawer()
{
}

LineRendererDrawer::~LineRendererDrawer()
{
}

void LineRendererDrawer::render(Clipboard &clipboard, const PhysicsEngine::Guid& id)
{
    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("LineRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        PhysicsEngine::LineRenderer *lineRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::LineRenderer>(id);

        if (lineRenderer != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            glm::vec3 start = lineRenderer->mStart;
            glm::vec3 end = lineRenderer->mEnd;

            if (ImGui::InputFloat3("Start", glm::value_ptr(start)))
            {
                lineRenderer->mStart = start;
            }
            if (ImGui::InputFloat3("End", glm::value_ptr(end)))
            {
                lineRenderer->mEnd = end;
            }

            bool enabled = lineRenderer->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                lineRenderer->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();

    if (isHovered())
    {
        if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
        {
            if (ImGui::MenuItem("RemoveComponent", NULL, false, true))
            {
                PhysicsEngine::LineRenderer* lineRenderer = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::LineRenderer>(id);
                clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(lineRenderer->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::LineRenderer>::type);
            }

            ImGui::EndPopup();
        }
    }
}

bool LineRendererDrawer::isHovered() const
{
    ImVec2 cursorPos = ImGui::GetMousePos();

    glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
    glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

    PhysicsEngine::Rect rect(min, max);

    return rect.contains(cursorPos.x, cursorPos.y);
}