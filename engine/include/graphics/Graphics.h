#ifndef __GRAPHICS_H__
#define __GRAPHICS_H__

#include "../glm/glm.hpp"

#include "../core/World.h"
#include "../core/Texture2D.h"
#include "../core/Texture3D.h"
#include "../core/Cubemap.h"
#include "../core/Shader.h"
#include "../core/Font.h"
#include "../core/Mesh.h"
#include "../core/Material.h"
#include "../core/Line.h"

#include "GraphicsState.h"
#include "GraphicsQuery.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	typedef struct GBuffer
	{
		GLenum gBufferStatus;
		GLuint handle;
		GLuint color0; // position
		GLuint color1; // normal
		GLuint color2; // color + spec
		GLuint depth;
		Shader shader;
	}GBuffer;

	typedef struct Framebuffer 
	{
		GLenum framebufferStatus;
		GLuint handle;
		Texture2D colorBuffer;
		Texture2D depthBuffer;
	}Framebuffer;

	typedef struct DebugWindow
	{
		float x;
		float y;
		float width;
		float height;

		GLuint VAO;
		GLuint vertexVBO;
		GLuint texCoordVBO;

		Shader shader;

		void init();
	}DebugWindow;


	typedef struct PerformanceGraph
	{
		float x;
		float y;
		float width;
		float height;
		float rangeMin;
		float rangeMax;
		float currentSample;
		int numberOfSamples;

		std::vector<float> samples;

		GLuint VAO;
		GLuint VBO;

		Shader shader;

		void init();
		void add(float sample);
	}PerformanceGraph;

	typedef struct LineBuffer 
	{
		size_t size;
		Shader shader;
		GLuint VAO;
		GLuint VBO;

		LineBuffer();
		~LineBuffer();
	}LineBuffer;

	typedef struct MeshBuffer
	{
		std::vector<Guid> meshIds;
		std::vector<int> start;  
		std::vector<int> count;
		std::vector<float> vertices; //could instead hold the mesh global index instead of duplicating vertices, normals, texcoords?? Or not hold them at all as they are just needed for filling out the mesh buffer vbo's??
		std::vector<float> normals;
		std::vector<float> texCoords;

		std::vector<Sphere> boundingSpheres;

		GLuint vao;
		GLuint vbo[3];

    	MeshBuffer();
    	~MeshBuffer();

    	int getIndex(Guid meshId);
	}MeshBuffer;


	class Graphics
	{
		public:
			//static GLHandle query;
			//static unsigned int gpu_time;

			//static void initializeGraphicsAPI();

			static void checkError();
			static void checkFrambufferError();
			static GLenum getTextureFormat(TextureFormat format);
			// static void enableBlend();
			// static void enableDepthTest();
			// static void enableCubemaps();
			// static void enablePoints();
			// static void setDepth(GLDepth depth);
			// static void setBlending(GLBlend src, GLBlend dest);
			// static void setViewport(int x, int y, int width, int height);
			// static void clearColorBuffer(glm::vec4 value);
			// static void clearDepthBuffer(float value);

			//static void beginGPUTimer();
			//static int endGPUTimer();

			// static void generate(GLFramebuffer* framebuffer);
			// static void bind(GLFramebuffer* framebuffer);
			// static void unbind(GLFramebuffer* framebuffer);


			static void readPixels(Texture2D* texture);
			static void apply(Texture2D* texture);
			// static void generate(Texture2D* texture);
			// static void destroy(Texture2D* texture);
			// static void bind(Texture2D* texture);
			// static void unbind(Texture2D* texture);
			// static void active(Texture2D* texture, unsigned int slot);

			static void readPixels(Texture3D* texture);
			static void apply(Texture3D* texture);
			// static void generate(Texture3D* texture);
			// static void destroy(Texture3D* texture);
			// static void bind(Texture3D* texture);
			// static void unbind(Texture3D* texture);
			// static void active(Texture3D* texture, unsigned int slot);

			static void readPixels(Cubemap* cubemap);
			static void apply(Cubemap* cubemap);
			// static void generate(Cubemap* cubemap);
			// static void destroy(Cubemap* cubemap);
			// static void bind(Cubemap* cubemap);
			// static void unbind(Cubemap* cubemap);
			
			// static void bind(World* world, Material* material, glm::mat4 model);
			// static void unbind(Material* material);

			static void use(GLuint shaderProgram, Material* material, RenderObject renderObject);

			static void compile(Shader* shader);
			static void use(Shader* shader, ShaderVariant variant);
			static void unuse(Shader* shader);
			static void setUniformBlock(Shader* shader, std::string blockName, int bindingPoint);
			static void setBool(Shader* shader, ShaderVariant variant, std::string name, bool value);
			static void setInt(Shader* shader, ShaderVariant variant, std::string name, int value);
			static void setFloat(Shader* shader, ShaderVariant variant, std::string name, float value);
			static void setVec2(Shader* shader, ShaderVariant variant, std::string name, glm::vec2 vec);
			static void setVec3(Shader* shader, ShaderVariant variant, std::string name, glm::vec3 vec);
			static void setVec4(Shader* shader, ShaderVariant variant, std::string name, glm::vec4 vec);
			static void setMat2(Shader* shader, ShaderVariant variant, std::string name, glm::mat2 mat);
			static void setMat3(Shader* shader, ShaderVariant variant, std::string name, glm::mat3 mat);
			static void setMat4(Shader* shader, ShaderVariant variant, std::string name, glm::mat4 mat);


			static void use(GLuint shaderProgram);
			static void setBool(GLuint shaderProgram, std::string name, bool value);
			static void setInt(GLuint shaderProgram, std::string name, int value);
			static void setFloat(GLuint shaderProgram, std::string name, float value);
			static void setVec2(GLuint shaderProgram, std::string name, glm::vec2 vec);
			static void setVec3(GLuint shaderProgram, std::string name, glm::vec3 vec);
			static void setVec4(GLuint shaderProgram, std::string name, glm::vec4 vec);
			static void setMat2(GLuint shaderProgram, std::string name, glm::mat2 mat);
			static void setMat3(GLuint shaderProgram, std::string name, glm::mat3 mat);
			static void setMat4(GLuint shaderProgram, std::string name, glm::mat4 mat);
			// static void setUniformBlockToBindingPoint(Shader* shader, std::string blockName, unsigned int bindingPoint);

			// static void apply(Line* line);
			// static void generate(Line* line);
			// static void destroy(Line* line);
			// static void bind(Line* line);
			// static void unbind(Line* line);
			// static void draw(Line* line);

			// static void apply(Mesh* mesh);
			// static void generate(Mesh* mesh);
			// static void destroy(Mesh* mesh);
			// static void bind(Mesh* mesh);
			// static void unbind(Mesh* mesh);
			// static void draw(Mesh* mesh);

			// static void apply(Boids* boids);
			// static void generate(Boids* boids);
			// static void destroy(Boids* boids);
			// static void bind(Boids* boids);
			// static void unbind(Boids* boids);
			// static void draw(Boids* boids);

			// static void apply(PerformanceGraph* graph);
			// static void generate(PerformanceGraph* graph);
			// static void destroy(PerformanceGraph* graph);
			// static void bind(PerformanceGraph* graph);
			// static void unbind(PerformanceGraph* graph);
			// static void draw(PerformanceGraph* graph);

			// static void apply(DebugWindow* window);
			// static void generate(DebugWindow* window);
			// static void destroy(DebugWindow* window);
			// static void bind(DebugWindow* window);
			// static void unbind(DebugWindow* window);
			// static void draw(DebugWindow* window);

			// static void apply(SlabNode* node);
			// static void generate(SlabNode* node);
			// static void destroy(SlabNode* node);
			// static void bind(SlabNode* node);
			// static void unbind(SlabNode* node);
			// static void draw(SlabNode* node);

			// static void generate(GLCamera* state);
			// static void destroy(GLCamera* state);
			// static void bind(GLCamera* state);
			// static void unbind(GLCamera* state);
			// static void setProjectionMatrix(GLCamera* state, glm::mat4 projection);
			// static void setViewMatrix(GLCamera* state, glm::mat4 view);
			// static void setCameraPosition(GLCamera* state, glm::vec3 position);

			// static void generate(GLShadow* state);
			// static void destroy(GLShadow* state);
			// static void bind(GLShadow* state);
			// static void unbind(GLShadow* state);
			// static void setLightProjectionMatrix(GLShadow* state, glm::mat4 projection, int index);
			// static void setLightViewMatrix(GLShadow* state, glm::mat4 view, int index);
			// static void setCascadeEnd(GLShadow* state, float cascadeEnd, int index);
			// static void setFarPlane(GLShadow* state, float farPlane);
			
			// static void generate(GLDirectionalLight* state);
			// static void destroy(GLDirectionalLight* state);
			// static void bind(GLDirectionalLight* state);
			// static void unbind(GLDirectionalLight* state);
			// static void setDirLightDirection(GLDirectionalLight* state, glm::vec3 direction);
			// static void setDirLightAmbient(GLDirectionalLight* state, glm::vec3 ambient);
			// static void setDirLightDiffuse(GLDirectionalLight* state, glm::vec3 diffuse);
			// static void setDirLightSpecular(GLDirectionalLight* state, glm::vec3 specular);
			
			// static void generate(GLSpotLight* state);
			// static void destroy(GLSpotLight* state);
			// static void bind(GLSpotLight* state);
			// static void unbind(GLSpotLight* state);
			// static void setSpotLightDirection(GLSpotLight* state, glm::vec3 direction);
			// static void setSpotLightPosition(GLSpotLight* state, glm::vec3 position);
			// static void setSpotLightAmbient(GLSpotLight* state, glm::vec3 ambient);
			// static void setSpotLightDiffuse(GLSpotLight* state, glm::vec3 diffuse);
			// static void setSpotLightSpecular(GLSpotLight* state, glm::vec3 specular);
			// static void setSpotLightConstant(GLSpotLight* state, float constant);
			// static void setSpotLightLinear(GLSpotLight* state, float linear);
			// static void setSpotLightQuadratic(GLSpotLight* state, float quadratic);
			// static void setSpotLightCutoff(GLSpotLight* state, float cutoff);
			// static void setSpotLightOuterCutoff(GLSpotLight* state, float cutoff);

			// static void generate(GLPointLight* state);
			// static void destroy(GLPointLight* state);
			// static void bind(GLPointLight* state);
			// static void unbind(GLPointLight* state);
			// static void setPointLightPosition(GLPointLight* state, glm::vec3 position);
			// static void setPointLightAmbient(GLPointLight* state, glm::vec3 ambient);
			// static void setPointLightDiffuse(GLPointLight* state, glm::vec3 diffuse);
			// static void setPointLightSpecular(GLPointLight* state, glm::vec3 specular);
			// static void setPointLightConstant(GLPointLight* state, float constant);
			// static void setPointLightLinear(GLPointLight* state, float linear);
			// static void setPointLightQuadratic(GLPointLight* state, float quadratic);


			static void render(World* world, Material* material, ShaderVariant variant, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query);
			static void render(World* world, Shader* shader, ShaderVariant variant, Texture2D* texture, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query);
			static void render(World* world, Shader* shader, ShaderVariant variant, glm::mat4 model, GLuint vao, GLenum mode, int numVertices, GraphicsQuery* query);
			static void renderText(World* world, Camera* camera, Font* font, std::string text, float x, float y, float scale, glm::vec3 color);


			//static void render(World* world, RenderObject renderObject, ShaderVariant variant, GraphicsQuery* query);
			// static void render(World* world, RenderObject renderObject, ShaderVariant variant, GLuint* shadowMaps, int shadowMapCount, GraphicsQuery* query);
			//static void render(World* world, RenderObject renderObject, GLuint shaderProgram, GraphicsQuery* query);
			//static void render(World* world, Shader* shader, ShaderVariant variant, glm::mat4 model, int start, GLsizei size, GraphicsQuery* query);
			// static void render(World* world, Shader* shader, ShaderVariant variant, glm::mat4 model, glm::mat4 view, glm::mat4 projection, int start, GLsizei size, GraphicsQuery* query);
			static void render(World* world, RenderObject renderObject, GraphicsQuery* query);
	};
}

#endif