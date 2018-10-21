#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Shader.h"
#include "Texture2D.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	typedef enum TEXURESLOT
	{
		MAINTEXTURE,
		NORMAL,
		DIFFUSE,
		SPECULAR,
		ALBEDO,
		GLOSS,
		CUBEMAP,
		COUNT
	}TEXTURESLOT;

	class Manager;

	class Material
	{
		public:
			int materialId;
			int shaderId;
			int textureId;
			int normalMapId;
			int specularMapId;

			float shininess;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;

		private:
			Manager* manager;

		public:
			Material();
			~Material();

			void setManager(Manager* manager);

			Shader* getShader();
			Texture2D* getMainTexture();
			Texture2D* getNormalMap();
			Texture2D* getSpecularMap();
	};
}

#endif