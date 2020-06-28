#ifndef __INTERNAL_SHADERS_H__
#define __INTERNAL_SHADERS_H__

#include <string>

namespace PhysicsEngine
{
	class InternalShaders
	{
		public:
			static const std::string lineVertexShader;
			static const std::string lineFragmentShader;
			static const std::string colorVertexShader;
			static const std::string colorFragmentShader;
			static const std::string graphVertexShader;
			static const std::string graphFragmentShader;
			static const std::string windowVertexShader;
			static const std::string windowFragmentShader;
			static const std::string normalMapVertexShader;
			static const std::string normalMapFragmentShader;
			static const std::string depthMapVertexShader;
			static const std::string depthMapFragmentShader;
			static const std::string shadowDepthMapVertexShader;
			static const std::string shadowDepthMapFragmentShader;
			static const std::string shadowDepthCubemapVertexShader;
			static const std::string shadowDepthCubemapGeometryShader;
			static const std::string shadowDepthCubemapFragmentShader;
			static const std::string overdrawVertexShader;
			static const std::string overdrawFragmentShader;
			static const std::string fontVertexShader;
			static const std::string fontFragmentShader;
			static const std::string instanceVertexShader;
			static const std::string instanceFragmentShader;
			static const std::string gbufferVertexShader;
			static const std::string gbufferFragmentShader;
			static const std::string positionAndNormalsVertexShader;
			static const std::string positionAndNormalsFragmentShader;
			static const std::string ssaoVertexShader;
			static const std::string ssaoFragmentShader;
			static const std::string simpleLitVertexShader;
			static const std::string simpleLitFragmentShader;
			static const std::string simpleLitDeferredVertexShader;
			static const std::string simpleLitDeferredFragmentShader;
	};
}

#endif