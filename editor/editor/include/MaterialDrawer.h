#ifndef __MATERIAL_DRAWER_H__
#define __MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"
#include "EditorClipboard.h"
#include "EditorProject.h"
//s#include "MaterialRenderer.h"

#include "core/World.h"
#include "core/Material.h"

#include "graphics/ForwardRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"
#include "../include/imgui_extensions.h"

namespace PhysicsEditor
{
	class MaterialDrawer : public InspectorDrawer
	{
		private:
			bool materialViewWorldPopulated;
			World materialViewWorld;

		public:
			MaterialDrawer();
			~MaterialDrawer();

			void render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id);

		private:
			void populateMaterialViewWorld(Material* material, Shader* shader);
	};



	template <GLenum T>
	struct UniformDrawer
	{
		static void draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project);
	};

	template<GLenum T>
	inline void UniformDrawer<T>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project) 
	{

	}

	template<>
	inline void UniformDrawer<GL_INT>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		int temp = material->getInt(uniform->name);

		if (ImGui::InputInt(uniform->shortName, &temp)) {
			material->setInt(uniform->name, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		float temp = material->getFloat(uniform->name);

		if (ImGui::InputFloat(uniform->shortName, &temp))
		{
			material->setFloat(uniform->name, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC2>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		glm::vec2 temp = material->getVec2(uniform->name);

		if (ImGui::InputFloat2(uniform->shortName, &temp[0]))
		{
			material->setVec2(uniform->name, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC3>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		glm::vec3 temp = material->getVec3(uniform->name);

		if (ImGui::InputFloat3(uniform->shortName, &temp[0]))
		{
			material->setVec3(uniform->name, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC4>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		glm::vec4 temp = material->getVec4(uniform->name);

		if (ImGui::InputFloat4(uniform->shortName, &temp[0]))
		{
			material->setVec4(uniform->name, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_SAMPLER_2D>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{
		Guid textureId = material->getTexture(uniform->name);

		Texture2D* texture = world->getAsset<Texture2D>(textureId);

		bool slotFilled = false;
		bool isClicked = ImGui::ImageSlot(uniform->shortName, texture == NULL ? 0 : texture->getNativeGraphics(), clipboard.getDraggedType() == InteractionType::Texture2D, &slotFilled);
		if (slotFilled) 
		{
			textureId = clipboard.getDraggedId();
			clipboard.clearDraggedItem();

			material->setTexture(uniform->name, textureId);

			project.isDirty = true;
		}
	}

	/*template<>
	void UniformDrawer<GL_SAMPLER_CUBE>::draw(World* world, Material* material, ShaderUniform* uniform, EditorClipboard& clipboard, EditorProject& project)
	{

	}*/
}

#endif