#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Guid.h"
#include "Asset.h"
#include "Shader.h"
#include "Texture2D.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MaterialHeader
	{
		Guid mAssetId;
		Guid mShaderId;
		size_t mUniformCount;
	};
#pragma pack(pop)

	class Material : public Asset
	{
		private:
			Guid mShaderId;
			bool mShaderChanged;
			std::vector<ShaderUniform> mUniforms;

		public:
			Material();
			Material(std::vector<char> data);
			~Material();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			void load(const std::string& filepath);
			void apply(World* world);
			void onShaderChanged(World* world);
			bool hasShaderChanged() const;

			void setShaderId(Guid shaderId);
			Guid getShaderId() const;
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

		private:
			int findIndexOfUniform(std::string name) const;
			int findIndexOfUniform(int nameLocation) const;
	};

	template <>
	const int AssetType<Material>::type = 4;

	template <typename T>
	struct IsMaterial { static const bool value; };

	template <typename T>
	const bool IsMaterial<T>::value = false;

	template<>
	const bool IsMaterial<Material>::value = true;
	template<>
	const bool IsAsset<Material>::value = true;
}

#endif