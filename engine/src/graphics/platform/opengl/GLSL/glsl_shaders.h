#ifndef GLSL_SHADERS_H__
#define GLSL_SHADERS_H__

#include <string>

namespace glsl
{
	std::string getGeometryVertexShader();
	std::string getGeometryFragmentShader();
    std::string getSSAOVertexShader();
    std::string getSSAOFragmentShader();
    std::string getShadowDepthMapVertexShader();
    std::string getShadowDepthMapFragmentShader();
    std::string getShadowDepthCubemapVertexShader();
    std::string getShadowDepthCubemapFragmentShader();
    std::string getShadowDepthCubemapGeometryShader();
    std::string getColorVertexShader();
    std::string getColorFragmentShader();
    std::string getColorInstancedVertexShader();
    std::string getColorInstancedFragmentShader();
    std::string getScreenQuadVertexShader();
    std::string getScreenQuadFragmentShader();
    std::string getSpriteVertexShader();
    std::string getSpriteFragmentShader();
    std::string getGBufferVertexShader();
    std::string getGBufferFragmentShader();
    std::string getNormalVertexShader();
    std::string getNormalFragmentShader();
    std::string getNormalInstancedVertexShader();
    std::string getNormalInstancedFragmentShader();
    std::string getPositionVertexShader();
    std::string getPositionFragmentShader();
    std::string getPositionInstancedVertexShader();
    std::string getPositionInstancedFragmentShader();
    std::string getLinearDepthVertexShader();
    std::string getLinearDepthFragmentShader();
    std::string getLinearDepthInstancedVertexShader();
    std::string getLinearDepthInstancedFragmentShader();
    std::string getLineVertexShader();
    std::string getLineFragmentShader();
    std::string getGizmoVertexShader();
    std::string getGizmoFragmentShader();
    std::string getGizmoInstancedVertexShader();
    std::string getGizmoInstancedFragmentShader();
    std::string getGridVertexShader();
    std::string getGridFragmentShader();
	std::string getStandardVertexShader();
	std::string getStandardFragmentShader();
}

#endif