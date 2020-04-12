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

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct ShaderHeader
	{
		Guid mShaderId;
		size_t mVertexShaderSize;
		size_t mGeometryShaderSize;
		size_t mFragmentShaderSize;
		size_t mNumberOfShaderUniforms;
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
		ShaderVersion mVersion;
		int mVariant;
		GLuint mHandle;
		bool mCompiled;
	};

	struct ShaderUniform
	{
		char mData[64];
		char mName[32]; //variable name in GLSL (including block name if applicable)
		char mShortName[32]; //variable name in GLSL (excluding block name if applicable)
		char mBlockName[32]; //block name (empty string if not part of block)
		size_t mNameLength; // length of name
		size_t mSize; // size of the uniform
		GLenum mType; // type of the uniform (float, vec3 or mat4, etc)
		int mVariant; // variant this uniform occurs in
		int mLocation; //uniform location in shader program
		size_t mIndex; // what index in array of uniforms we are at
	};

	struct ShaderAttribute
	{
		char mName[32];
	};

	class Shader : public Asset
	{
		private:
			std::string mVertexShader;
			std::string mFragmentShader;
			std::string mGeometryShader;

			bool mAllProgramsCompiled;
			int mActiveProgram;
			std::vector<ShaderProgram> mPrograms;
			std::vector<ShaderUniform> mUniforms; 
			std::vector<ShaderAttribute> mAttributes;

		public:
			Shader();
			Shader(std::vector<char> data);
			~Shader();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			void load(const std::string& filepath);
			void load(const std::string vertexShader, const std::string fragmentShader, const std::string geometryShader);

			bool isCompiled() const;
			bool contains(int variant) const;
			void add(int variant);
			void remove(int variant);

			void compile();
			void use(int program);
			void unuse();
			void setVertexShader(const std::string vertexShader);
			void setGeometryShader(const std::string geometryShader);
			void setFragmentShader(const std::string fragmentShader);
			void setUniformBlock(const std::string& blockName, int bindingPoint) const;
			int findUniformLocation(const std::string& name, int program) const;
			int getProgramFromVariant(int variant) const;

			std::vector<ShaderProgram> getPrograms() const;
			std::vector<ShaderUniform> getUniforms() const;
			std::vector<ShaderAttribute> getAttributeNames() const;
			std::string getVertexShader() const;
			std::string getGeometryShader() const;
			std::string getFragmentShader() const;

			void setBool(const char* name, bool value) const;
			void setInt(const char* name, int value) const;
			void setFloat(const char* name, float value) const;
			void setVec2(const char* name, const glm::vec2& vec) const;
			void setVec3(const char* name, const glm::vec3& vec) const;
			void setVec4(const char* name, const glm::vec4& vec) const;
			void setMat2(const char* name, const glm::mat2& mat) const;
			void setMat3(const char* name, const glm::mat3& mat) const;
			void setMat4(const char* name, const glm::mat4& mat) const;

			void setBool(int nameLocation, bool value) const;
			void setInt(int nameLocation, int value) const;
			void setFloat(int nameLocation, float value) const;
			void setVec2(int nameLocation, const glm::vec2 &vec) const;
			void setVec3(int nameLocation, const glm::vec3 &vec) const;
			void setVec4(int nameLocation, const glm::vec4 &vec) const;
			void setMat2(int nameLocation, const glm::mat2 &mat) const;
			void setMat3(int nameLocation, const glm::mat3 &mat) const;
			void setMat4(int nameLocation, const glm::mat4 &mat) const;

			bool getBool(const char* name) const;
			int getInt(const char* name) const;
			float getFloat(const char* name) const;
			glm::vec2 getVec2(const char* name) const;
			glm::vec3 getVec3(const char* name) const;
			glm::vec4 getVec4(const char* name) const;
			glm::mat2 getMat2(const char* name) const;
			glm::mat3 getMat3(const char* name) const;
			glm::mat4 getMat4(const char* name) const;

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

	template <typename T>
	struct IsShader { static const bool value; };

	template <typename T>
	const bool IsShader<T>::value = false;

	template<>
	const bool IsShader<Shader>::value = true;
	template<>
	const bool IsAsset<Shader>::value = true;
}

#endif