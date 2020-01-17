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

		private:
			Shader* shader;

		public:
			Material();
			Material(std::vector<char> data);
			~Material();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void use(Shader* shader, RenderObject renderObject);

			//Guid getShaderId() const;
			//void setShaderId(World* world, Guid id); // have this call onShaderChanged(world)?

			void onShaderChanged(World* world); // sets shader pointer and fills uniform data?
			//void load(World* world);
			//void loadShaderFromWorld(World* world);
			//void printUniforms(Shader* shader);


			void setBool(std::string name, bool value) const;
			void setInt(std::string name, int value) const;
			void setFloat(std::string name, float value) const;
			void setVec2(std::string name, const glm::vec2& vec) const;
			void setVec3(std::string name, const glm::vec3& vec) const;
			void setVec4(std::string name, const glm::vec4& vec) const;
			void setMat2(std::string name, const glm::mat2& mat) const;
			void setMat3(std::string name, const glm::mat3& mat) const;
			void setMat4(std::string name, const glm::mat4& mat) const;

			void setBool(int nameLocation, bool value) const;
			void setInt(int nameLocation, int value) const;
			void setFloat(int nameLocation, float value) const;
			void setVec2(int nameLocation, const glm::vec2& vec) const;
			void setVec3(int nameLocation, const glm::vec3& vec) const;
			void setVec4(int nameLocation, const glm::vec4& vec) const;
			void setMat2(int nameLocation, const glm::mat2& mat) const;
			void setMat3(int nameLocation, const glm::mat3& mat) const;
			void setMat4(int nameLocation, const glm::mat4& mat) const;

			bool getBool(std::string name) const;
			int getInt(std::string name) const;
			float getFloat(std::string name) const;
			glm::vec2 getVec2(std::string name) const;
			glm::vec3 getVec3(std::string name) const;
			glm::vec4 getVec4(std::string name) const;
			glm::mat2 getMat2(std::string name) const;
			glm::mat3 getMat3(std::string name) const;
			glm::mat4 getMat4(std::string name) const;

			bool getBool(int nameLocation) const;
			int getInt(int nameLocation) const;
			float getFloat(int nameLocation) const;
			glm::vec2 getVec2(int nameLocation) const;
			glm::vec3 getVec3(int nameLocation) const;
			glm::vec4 getVec4(int nameLocation) const;
			glm::mat2 getMat2(int nameLocation) const;
			glm::mat3 getMat3(int nameLocation) const;
			glm::mat4 getMat4(int nameLocation) const;
	};

	template <>
	const int AssetType<Material>::type = 4;
}

#endif