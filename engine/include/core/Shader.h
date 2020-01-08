#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "../glm/glm.hpp"

#include "Guid.h"
#include "Asset.h"

#include "../graphics/GraphicsHandle.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct ShaderHeader
	{
		Guid shaderId;
		size_t vertexShaderSize;
		size_t geometryShaderSize;
		size_t fragmentShaderSize;
	};
#pragma pack(pop)


	enum ShaderVariant
	{
		None = 0,
		Directional = 1,
		Spot = 2, 
		Point = 4,
		HardShadows = 8,
		SoftShadows = 16,
		SSAO = 32,
		Cascade = 64
	};

	enum ShaderVersion
	{
		GL330,
		GL430
	};

	struct ShaderProgram
	{
		ShaderVersion version;
		int variant;
		GLuint handle;
		bool compiled;
	};

	class Shader : public Asset
	{
		public:
			std::string vertexShader;
			std::string fragmentShader;
			std::string geometryShader;

		private:
			bool allCompiled;
			int activeProgramIndex;
			std::vector<ShaderProgram> programs;

		public:
			Shader();
			Shader(std::vector<char> data);
			~Shader();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool isCompiled() const;
			bool contains(int variant) const;
			void add(int variant);
			void remove(int variant);

			void compile();
			void use(int variant);
			void unuse();
			void setUniformBlock(std::string blockName, int bindingPoint) const;
			int findUniformLocation(std::string name) const;

			void setBool(std::string name, bool value) const;
			void setInt(std::string name, int value) const;
			void setFloat(std::string name, float value) const;
			void setVec2(std::string name, const glm::vec2 &vec) const;
			void setVec3(std::string name, const glm::vec3 &vec) const;
			void setVec4(std::string name, const glm::vec4 &vec) const;
			void setMat2(std::string name, const glm::mat2 &mat) const;
			void setMat3(std::string name, const glm::mat3 &mat) const;
			void setMat4(std::string name, const glm::mat4 &mat) const;

			void setBool(int nameLocation, bool value) const;
			void setInt(int nameLocation, int value) const;
			void setFloat(int nameLocation, float value) const;
			void setVec2(int nameLocation, const glm::vec2 &vec) const;
			void setVec3(int nameLocation, const glm::vec3 &vec) const;
			void setVec4(int nameLocation, const glm::vec4 &vec) const;
			void setMat2(int nameLocation, const glm::mat2 &mat) const;
			void setMat3(int nameLocation, const glm::mat3 &mat) const;
			void setMat4(int nameLocation, const glm::mat4 &mat) const;

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
	const int AssetType<Shader>::type = 0;
}

#endif