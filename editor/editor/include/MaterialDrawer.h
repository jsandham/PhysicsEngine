#ifndef __MATERIAL_DRAWER_H__
#define __MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"
#include "EditorClipboard.h"
#include "EditorProject.h"

#include "core/World.h"
#include "core/Material.h"
#include "components/MeshRenderer.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include "../include/imgui_extensions.h"

namespace PhysicsEditor
{
	class MaterialDrawer : public InspectorDrawer
	{
		private:
			World previewWorld; 
			Camera* previewCamera;
			MeshRenderer* sphereMeshRenderer;
			//Shader* previewShader;
			//Material* previewMaterial;

		public:
			MaterialDrawer();
			~MaterialDrawer();

			void render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id);

		private:
			void populatePreviewWorld(World* world);



			//void initViewWorld();
			//void updateViewWorld();
			//void drawUniforms();
			//void drawPreview();
	};



	template <GLenum T>
	struct UniformDrawer
	{
		static void draw(World* world, 
						 Material* material,
						 ShaderUniform* uniform, 
						 EditorClipboard& clipboard, 
						 EditorProject& project);
	};

	template<GLenum T>
	inline void UniformDrawer<T>::draw(World* world, 
									   Material* material, 
									   ShaderUniform* uniform, 
									   EditorClipboard& clipboard, 
									   EditorProject& project)
	{

	}

	template<>
	inline void UniformDrawer<GL_INT>::draw(World* world, 
										    Material* material, 
											ShaderUniform* uniform, 
											EditorClipboard& clipboard, 
											EditorProject& project)
	{
		int temp = material->getInt(uniform->mName);

		if (ImGui::InputInt(uniform->mShortName, &temp)) {
			material->setInt(uniform->mName, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT>::draw(World* world, 
											  Material* material,
											  ShaderUniform* uniform, 
											  EditorClipboard& clipboard, 
											  EditorProject& project)
	{
		float temp = material->getFloat(uniform->mName);

		if (ImGui::InputFloat(uniform->mShortName, &temp))
		{
			material->setFloat(uniform->mName, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC2>::draw(World* world, 
												   Material* material,
												   ShaderUniform* uniform, 
												   EditorClipboard& clipboard, 
												   EditorProject& project)
	{
		glm::vec2 temp = material->getVec2(uniform->mName);

		if (ImGui::InputFloat2(uniform->mShortName, &temp[0]))
		{
			material->setVec2(uniform->mName, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC3>::draw(World* world, 
												   Material* material,
												   ShaderUniform* uniform, 
												   EditorClipboard& clipboard, 
												   EditorProject& project)
	{
		glm::vec3 temp = material->getVec3(uniform->mName);

		if (ImGui::InputFloat3(uniform->mShortName, &temp[0]))
		{
			material->setVec3(uniform->mName, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_FLOAT_VEC4>::draw(World* world, 
												   Material* material, 
												   ShaderUniform* uniform, 
												   EditorClipboard& clipboard, 
												   EditorProject& project)
	{
		glm::vec4 temp = material->getVec4(uniform->mName);

		if (ImGui::InputFloat4(uniform->mShortName, &temp[0]))
		{
			material->setVec4(uniform->mName, temp);
			project.isDirty = true;
		}
	}

	template<>
	inline void UniformDrawer<GL_SAMPLER_2D>::draw(World* world, 
												   Material* material,
												   ShaderUniform* uniform, 
												   EditorClipboard& clipboard, 
												   EditorProject& project)
	{
		Guid textureId = material->getTexture(uniform->mName);

		Texture2D* texture = world->getAsset<Texture2D>(textureId);

		bool slotFilled = false;
		bool isClicked = ImGui::ImageSlot(uniform->mShortName, texture == NULL ? 0 : texture->getNativeGraphics(), clipboard.getDraggedType() == InteractionType::Texture2D, &slotFilled);
		if (slotFilled) 
		{
			textureId = clipboard.getDraggedId();
			clipboard.clearDraggedItem();

			material->setTexture(uniform->mName, textureId);

			project.isDirty = true;
		}
	}
}

#endif