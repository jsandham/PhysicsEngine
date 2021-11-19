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

void LineRendererDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("LineRenderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        LineRenderer *lineRenderer = clipboard.getWorld()->getComponentById<LineRenderer>(id);

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
}