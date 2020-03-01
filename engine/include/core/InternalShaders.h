#ifndef __INTERNAL_SHADERS_H__
#define __INTERNAL_SHADERS_H__

#include <string>

namespace PhysicsEngine
{
	class InternalShaders
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
			static std::string positionAndNormalsVertexShader;
			static std::string positionAndNormalsFragmentShader;
			static std::string ssaoVertexShader;
			static std::string ssaoFragmentShader;
			static std::string simpleLitVertexShader;
			static std::string simpleLitFragmentShader;
	};
}

#endif