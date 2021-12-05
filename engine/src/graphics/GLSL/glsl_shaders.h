#ifndef GLSL_SHADERS_H__
#define GLSL_SHADERS_H__

namespace PhysicsEngine
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
    std::string getScreenQuadVertexShader();
    std::string getScreenQuadFragmentShader();
    std::string getSpriteVertexShader();
    std::string getSpriteFragmentShader();
    std::string getGBufferVertexShader();
    std::string getGBufferFragmentShader();
    std::string getNormalVertexShader();
    std::string getNormalFragmentShader();
    std::string getPositionVertexShader();
    std::string getPositionFragmentShader();
    std::string getLinearDepthVertexShader();
    std::string getLinearDepthFragmentShader();
    std::string getLineVertexShader();
    std::string getLineFragmentShader();
    std::string getGizmoVertexShader();
    std::string getGizmoFragmentShader();
    std::string getGridVertexShader();
    std::string getGridFragmentShader();
	std::string getStandardVertexShader();
	std::string getStandardFragmentShader();
}

#endif