#ifndef __SHADER_H__
#define __SHADER_H__

#define NOMINMAX

#include <string>
#include <vector>

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
		Directional,
		Directional_Hard,
		Directional_Soft,
		Spot,
		Spot_Hard,
		Spot_Soft,
		Point,
		Point_Hard,
		Point_Soft,
		None
	};
	
	class Shader : public Asset
	{
		public:
			static std::string lineVertexShader;
			static std::string lineFragmentShader;
			static std::string colorVertexShader;
			static std::string colorFragmentShader;
			static std::string graphVertexShader;
			static std::string graphFragmentShader;
			static std::string windowVertexShader;
			static std::string windowFragmentShader;
			static std::string normalMapVertexShader;
			static std::string normalMapFragmentShader;
			static std::string depthMapVertexShader;
			static std::string depthMapFragmentShader;
			static std::string shadowDepthMapVertexShader;
			static std::string shadowDepthMapFragmentShader;
			static std::string shadowDepthCubemapVertexShader;
			static std::string shadowDepthCubemapGeometryShader;
			static std::string shadowDepthCubemapFragmentShader;
			static std::string overdrawVertexShader;
			static std::string overdrawFragmentShader;
			static std::string fontVertexShader;
			static std::string fontFragmentShader;
			static std::string instanceVertexShader;
			static std::string instanceFragmentShader;
			static std::string gbufferVertexShader;
			static std::string gbufferFragmentShader;
			static std::string mainVertexShader;
			static std::string mainFragmentShader;
			static std::string ssaoVertexShader;
			static std::string ssaoFragmentShader;

		public:
			std::string vertexShader;
			std::string fragmentShader;
			std::string geometryShader;

			bool programCompiled;

			GraphicsHandle programs[10]; // could call this variants??

		public:
			Shader();
			Shader(std::vector<char> data);
			~Shader();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool isCompiled();
			void compile();
			void setUniformBlock(std::string blockName, int bindingPoint);

			void setBool(std::string name, ShaderVariant variant, bool value);
			void setInt(std::string name, ShaderVariant variant, int value);
			void setFloat(std::string name, ShaderVariant variant, float value);
			void setVec2(std::string name, ShaderVariant variant, glm::vec2 &vec);
			void setVec3(std::string name, ShaderVariant variant, glm::vec3 &vec);
			void setVec4(std::string name, ShaderVariant variant, glm::vec4 &vec);
			void setMat2(std::string name, ShaderVariant variant, glm::mat2 &mat);
			void setMat3(std::string name, ShaderVariant variant, glm::mat3 &mat);
			void setMat4(std::string name, ShaderVariant variant, glm::mat4 &mat);
	};

	template <>
	const int AssetType<Shader>::type = 0;
}

#endif