#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Guid.h"
#include "Asset.h"
#include "Shader.h"
#include "Texture2D.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../../include/graphics/RenderObject.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MaterialHeader
	{
		Guid assetId;
		Guid shaderId;
		Guid textureId;
		Guid normalMapId;
		Guid specularMapId;

		float shininess;
		glm::vec3 ambient;
		glm::vec3 diffuse;
		glm::vec3 specular;
		glm::vec4 color;
	};
#pragma pack(pop)

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

	class Material : public Asset
	{
		public:
			Guid shaderId;
			Guid textureId;
			Guid normalMapId;
			Guid specularMapId;

			float shininess;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;
			glm::vec4 color;

		public:
			Material();
			Material(std::vector<char> data);
			~Material();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void use(Shader* shader, RenderObject renderObject);
	};

	template <>
	const int AssetType<Material>::type = 4;
}

#endif