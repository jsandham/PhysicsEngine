#include "../include/MaterialDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{

}

MaterialDrawer::~MaterialDrawer()
{

}

void MaterialDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	Material* material = world->getAsset<Material>(id);

	Guid currentShaderId = material->getShaderId();

	// dropdown for selecting shader for material
	if (ImGui::BeginCombo("Shader", currentShaderId.toString().c_str(), ImGuiComboFlags_None))
	{
		for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
			Shader* s = world->getAssetByIndex<Shader>(i);
			
			bool is_selected = (currentShaderId == s->assetId);
			if (ImGui::Selectable(s->assetId.toString().c_str(), is_selected)) {
				currentShaderId = s->assetId;

				material->setShaderId(currentShaderId);
				material->onShaderChanged(world);
			}
			if (is_selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

	// draw material uniforms
	std::vector<ShaderUniform> uniforms = material->getUniforms();
    for(size_t i = 0; i < uniforms.size(); i++)
	{
		// only expose uniforms exist in a Material uniform struct in the shader
		if (std::strcmp(uniforms[i].blockName, "material") != 0)
		{
			continue;
		}

		// Note: matrices not supported
		switch (uniforms[i].type)
		{
			case GL_INT:
				UniformDrawer<GL_INT>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT:
				UniformDrawer<GL_FLOAT>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC2:
				UniformDrawer<GL_FLOAT_VEC2>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC3:
				UniformDrawer<GL_FLOAT_VEC3>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC4:
				UniformDrawer<GL_FLOAT_VEC4>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_SAMPLER_2D:
				UniformDrawer<GL_SAMPLER_2D>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_SAMPLER_CUBE:
				UniformDrawer<GL_SAMPLER_CUBE>::draw(world, material, &uniforms[i], clipboard, project);
				break;

		}
    }

	ImGui::Separator();

	ImGui::Text("Preview");

	// Draw material preview child window
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
		ImGui::BeginChild("MaterialPreviewWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);

		//ImGui::Image((void*)(intptr_t)texture->handle.handle, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));
		// Call simple material renderer here to display material on a sphere

		// steps:
		// create frame buffer with color and depth attachment in initialization
		// bind framebuffer 
		// draw sphere with material and simple light from a fixed camera looking at sphere.
		// unbind frame buffer
		// take color texture from framebuffer and use it with ImGui::Image()

		ImGui::EndChild();
	}
}