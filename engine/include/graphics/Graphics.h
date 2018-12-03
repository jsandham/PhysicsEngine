#ifndef __GRAPHICS_H__
#define __GRAPHICS_H__

#include "../glm/glm.hpp"

#include "../core/Texture2D.h"
#include "../core/Texture3D.h"
#include "../core/Cubemap.h"
#include "../core/Shader.h"
#include "../core/Mesh.h"
#include "../core/Material.h"
#include "../core/Line.h"
#include "../core/PerformanceGraph.h"
#include "../core/DebugWindow.h"

#include "../graphics/GLFramebuffer.h"
#include "../graphics/GLState.h"
#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	typedef enum GLDepth
	{
		Never,
		Less,
		Equal,
		LEqual,
		Greater,
		NotEqual,
		GEqual,
		Always
	};

	typedef enum GLBlend
	{
		Zero,
		One, 
	};

	class Graphics
	{
		public:
			static GLHandle query;
			static unsigned int gpu_time;

			//static void initializeGraphicsAPI();

			static void checkError();
			static void enableBlend();
			static void enableDepthTest();
			static void enableCubemaps();
			static void enablePoints();
			static void setDepth(GLDepth depth);
			static void setBlending(GLBlend src, GLBlend dest);
			static void setViewport(int x, int y, int width, int height);
			static void clearColorBuffer(glm::vec4 value);
			static void clearDepthBuffer(float value);

			static void beginGPUTimer();
			static int endGPUTimer();

			static void generate(GLFramebuffer* framebuffer);
			static void bind(GLFramebuffer* framebuffer);
			static void unbind(GLFramebuffer* framebuffer);


			static void readPixels(Texture2D* texture);
			static void apply(Texture2D* texture);
			static void generate(Texture2D* texture);
			static void destroy(Texture2D* texture);
			static void bind(Texture2D* texture);
			static void unbind(Texture2D* texture);
			static void active(Texture2D* texture, unsigned int slot);

			static void readPixels(Texture3D* texture);
			static void apply(Texture3D* texture);
			static void generate(Texture3D* texture);
			static void destroy(Texture3D* texture);
			static void bind(Texture3D* texture);
			static void unbind(Texture3D* texture);
			static void active(Texture3D* texture, unsigned int slot);

			static void readPixels(Cubemap* cubemap);
			static void apply(Cubemap* cubemap);
			static void generate(Cubemap* cubemap);
			static void destroy(Cubemap* cubemap);
			static void bind(Cubemap* cubemap);
			static void unbind(Cubemap* cubemap);
			
			static void bind(Material* material, glm::mat4 model);
			static void unbind(Material* material);

			static void compile(Shader* shader);
			static void use(Shader* shader);
			static void unuse(Shader* shader);
			static void setBool(Shader* shader, std::string name, bool value);
			static void setInt(Shader* shader, std::string name, int value);
			static void setFloat(Shader* shader, std::string name, float value);
			static void setVec2(Shader* shader, std::string name, glm::vec2 &vec);
			static void setVec3(Shader* shader, std::string name, glm::vec3 &vec);
			static void setVec4(Shader* shader, std::string name, glm::vec4 &vec);
			static void setMat2(Shader* shader, std::string name, glm::mat2 &mat);
			static void setMat3(Shader* shader, std::string name, glm::mat3 &mat);
			static void setMat4(Shader* shader, std::string name, glm::mat4 &mat);
			static void setUniformBlockToBindingPoint(Shader* shader, std::string blockName, unsigned int bindingPoint);

			static void apply(Line* line);
			static void generate(Line* line);
			static void destroy(Line* line);
			static void bind(Line* line);
			static void unbind(Line* line);
			static void draw(Line* line);

			static void apply(Mesh* mesh);
			static void generate(Mesh* mesh);
			static void destroy(Mesh* mesh);
			static void bind(Mesh* mesh);
			static void unbind(Mesh* mesh);
			static void draw(Mesh* mesh);

			static void apply(PerformanceGraph* graph);
			static void generate(PerformanceGraph* graph);
			static void destroy(PerformanceGraph* graph);
			static void bind(PerformanceGraph* graph);
			static void unbind(PerformanceGraph* graph);
			static void draw(PerformanceGraph* graph);

			static void apply(DebugWindow* window);
			static void generate(DebugWindow* window);
			static void destroy(DebugWindow* window);
			static void bind(DebugWindow* window);
			static void unbind(DebugWindow* window);
			static void draw(DebugWindow* window);

			static void generate(GLCamera* state);
			static void destroy(GLCamera* state);
			static void bind(GLCamera* state);
			static void unbind(GLCamera* state);
			static void setProjectionMatrix(GLCamera* state, glm::mat4 projection);
			static void setViewMatrix(GLCamera* state, glm::mat4 view);
			static void setCameraPosition(GLCamera* state, glm::vec3 position);

			static void generate(GLShadow* state);
			static void destroy(GLShadow* state);
			static void bind(GLShadow* state);
			static void unbind(GLShadow* state);
			static void setLightProjectionMatrix(GLShadow* state, glm::mat4 projection, int index);
			static void setLightViewMatrix(GLShadow* state, glm::mat4 view, int index);
			static void setCascadeEnd(GLShadow* state, float cascadeEnd, int index);
			static void setFarPlane(GLShadow* state, float farPlane);
			
			static void generate(GLDirectionalLight* state);
			static void destroy(GLDirectionalLight* state);
			static void bind(GLDirectionalLight* state);
			static void unbind(GLDirectionalLight* state);
			static void setDirLightDirection(GLDirectionalLight* state, glm::vec3 direction);
			static void setDirLightAmbient(GLDirectionalLight* state, glm::vec3 ambient);
			static void setDirLightDiffuse(GLDirectionalLight* state, glm::vec3 diffuse);
			static void setDirLightSpecular(GLDirectionalLight* state, glm::vec3 specular);
			
			static void generate(GLSpotLight* state);
			static void destroy(GLSpotLight* state);
			static void bind(GLSpotLight* state);
			static void unbind(GLSpotLight* state);
			static void setSpotLightDirection(GLSpotLight* state, glm::vec3 direction);
			static void setSpotLightPosition(GLSpotLight* state, glm::vec3 position);
			static void setSpotLightAmbient(GLSpotLight* state, glm::vec3 ambient);
			static void setSpotLightDiffuse(GLSpotLight* state, glm::vec3 diffuse);
			static void setSpotLightSpecular(GLSpotLight* state, glm::vec3 specular);
			static void setSpotLightConstant(GLSpotLight* state, float constant);
			static void setSpotLightLinear(GLSpotLight* state, float linear);
			static void setSpotLightQuadratic(GLSpotLight* state, float quadratic);
			static void setSpotLightCutoff(GLSpotLight* state, float cutoff);
			static void setSpotLightOuterCutoff(GLSpotLight* state, float cutoff);

			static void generate(GLPointLight* state);
			static void destroy(GLPointLight* state);
			static void bind(GLPointLight* state);
			static void unbind(GLPointLight* state);
			static void setPointLightPosition(GLPointLight* state, glm::vec3 position);
			static void setPointLightAmbient(GLPointLight* state, glm::vec3 ambient);
			static void setPointLightDiffuse(GLPointLight* state, glm::vec3 diffuse);
			static void setPointLightSpecular(GLPointLight* state, glm::vec3 specular);
			static void setPointLightConstant(GLPointLight* state, float constant);
			static void setPointLightLinear(GLPointLight* state, float linear);
			static void setPointLightQuadratic(GLPointLight* state, float quadratic);
	};
}

#endif