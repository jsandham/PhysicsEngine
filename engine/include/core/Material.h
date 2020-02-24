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
		size_t uniformCount;
		/*Guid textureId;
		Guid normalMapId;
		Guid specularMapId;

		float shininess;
		glm::vec3 ambient;
		glm::vec3 diffuse;
		glm::vec3 specular;
		glm::vec4 color;*/
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

		private:
			bool shaderChanged;
			std::vector<ShaderUniform> uniforms;

		public:
			Material();
			Material(std::vector<char> data);
			~Material();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);
			void apply(World* world);
			void onShaderChanged(World* world);
			bool hasShaderChanged() const;

			//Guid getShaderId() const;
			//void setShaderId(World* world, Guid id); // have this call onShaderChanged(world)?

			//void onShaderChanged(World* world); // sets shader pointer and fills uniform data?
			//void load(World* world);
			//void loadShaderFromWorld(World* world);
			//void printUniforms(Shader* shader);

			std::vector<ShaderUniform> getUniforms() const;


			void setBool(std::string name, bool value);
			void setInt(std::string name, int value);
			void setFloat(std::string name, float value);
			void setColor(std::string name, const Color& color);
			void setVec2(std::string name, const glm::vec2& vec);
			void setVec3(std::string name, const glm::vec3& vec);
			void setVec4(std::string name, const glm::vec4& vec);
			void setMat2(std::string name, const glm::mat2& mat);
			void setMat3(std::string name, const glm::mat3& mat);
			void setMat4(std::string name, const glm::mat4& mat);
			void setTexture(std::string name, const Guid& textureId);

			void setBool(int nameLocation, bool value);
			void setInt(int nameLocation, int value);
			void setFloat(int nameLocation, float value);
			void setColor(int nameLocation, const Color& color);
			void setVec2(int nameLocation, const glm::vec2& vec);
			void setVec3(int nameLocation, const glm::vec3& vec);
			void setVec4(int nameLocation, const glm::vec4& vec);
			void setMat2(int nameLocation, const glm::mat2& mat);
			void setMat3(int nameLocation, const glm::mat3& mat);
			void setMat4(int nameLocation, const glm::mat4& mat);
			void setTexture(int nameLocation, const Guid& textureId);

			bool getBool(std::string name) const;
			int getInt(std::string name) const;
			float getFloat(std::string name) const;
			Color getColor(std::string name) const;
			glm::vec2 getVec2(std::string name) const;
			glm::vec3 getVec3(std::string name) const;
			glm::vec4 getVec4(std::string name) const;
			glm::mat2 getMat2(std::string name) const;
			glm::mat3 getMat3(std::string name) const;
			glm::mat4 getMat4(std::string name) const;
			Guid getTexture(std::string name) const;

			bool getBool(int nameLocation) const;
			int getInt(int nameLocation) const;
			float getFloat(int nameLocation) const;
			Color getColor(int nameLocation) const;
			glm::vec2 getVec2(int nameLocation) const;
			glm::vec3 getVec3(int nameLocation) const;
			glm::vec4 getVec4(int nameLocation) const;
			glm::mat2 getMat2(int nameLocation) const;
			glm::mat3 getMat3(int nameLocation) const;
			glm::mat4 getMat4(int nameLocation) const;
			Guid getTexture(int nameLocation) const;


			//This only exists so that we can set the uniforms in a dummy Material for the purpose of then calling the serialize method 
			//to get the serialized data when writing the material to binary. Used from AssetLoader. Think of a better way?
			void setUniformsEditorOnly(std::vector<ShaderUniform> uniforms);

		private:
			int findIndexOfUniform(std::string name) const;
			int findIndexOfUniform(int nameLocation) const;
	};

	template <>
	const int AssetType<Material>::type = 4;

	template <typename T>
	struct IsMaterial { static bool value; };

	template <typename T>
	bool IsMaterial<T>::value = false;

	template<>
	bool IsMaterial<Material>::value = true;
	template<>
	bool IsAsset<Material>::value = true;
}

#endif