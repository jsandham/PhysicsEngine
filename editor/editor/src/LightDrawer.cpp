#include "../include/LightDrawer.h"

#include "components/Light.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

LightDrawer::LightDrawer()
{

}

LightDrawer::~LightDrawer()
{

}

void LightDrawer::render(World world, Guid entityId, Guid componentId)
{
	if(ImGui::TreeNode("Light"))
	{
		Light* light = world.getComponentById<Light>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		float position[3];
		position[0] = light->position.x;
		position[1] = light->position.y;
		position[2] = light->position.z;

		float direction[3];
		direction[0] = light->direction.x;
		direction[1] = light->direction.y;
		direction[2] = light->direction.z;

		float ambient[3];
		ambient[0] = light->ambient.x;
		ambient[1] = light->ambient.y;
		ambient[2] = light->ambient.z;

		float diffuse[3];
		diffuse[0] = light->diffuse.x;
		diffuse[1] = light->diffuse.y;
		diffuse[2] = light->diffuse.z;

		float specular[3];
		specular[0] = light->specular.x;
		specular[1] = light->specular.y;
		specular[2] = light->specular.z;

		ImGui::InputFloat3("Position", &position[0]);
		ImGui::InputFloat3("Direction", &direction[0]);
		ImGui::InputFloat3("Ambient", &ambient[0]);
		ImGui::InputFloat3("Diffuse", &diffuse[0]);
		ImGui::InputFloat3("Specular", &specular[0]);

		light->position.x = position[0];
		light->position.y = position[1];
		light->position.z = position[2];

		light->direction.x = direction[0];
		light->direction.y = direction[1];
		light->direction.z = direction[2];

		light->ambient.x = ambient[0];
		light->ambient.y = ambient[1];
		light->ambient.z = ambient[2];

		light->diffuse.x = diffuse[0];
		light->diffuse.y = diffuse[1];
		light->diffuse.z = diffuse[2];

		light->specular.x = specular[0];
		light->specular.y = specular[1];
		light->specular.z = specular[2];

		ImGui::InputFloat("Constant", &light->constant);
		ImGui::InputFloat("Linear", &light->linear);
		ImGui::InputFloat("Quadratic", &light->quadratic);
		ImGui::InputFloat("Cut-Off", &light->cutOff);
		ImGui::InputFloat("Outer Cut-Off", &light->outerCutOff);

		const char* lightTypes[] = { "Directional", "Spot", "Point" };
		int lightTypeIndex = static_cast<int>(light->lightType);
		ImGui::Combo("##LightType", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes));
		light->lightType = static_cast<LightType>(lightTypeIndex);

		const char* shadowTypes[] = { "Hard", "Soft" };
		int shadowTypeIndex = static_cast<int>(light->shadowType);
		ImGui::Combo("##ShadowType", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes));
		light->shadowType = static_cast<ShadowType>(shadowTypeIndex);

		ImGui::TreePop();
	}
}