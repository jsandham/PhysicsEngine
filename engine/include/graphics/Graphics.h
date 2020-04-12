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

#include "GraphicsState.h"
#include "GraphicsQuery.h"
#include "RenderObject.h"

#define LOG 0

//#if LOG 
//	#define GLLOG(func) (printf("%s\n", #func)); func  
//#else
//	#define GLLOG(func) func
//#endif

constexpr std::uint32_t stringLength(const char* cstr)
{
	return (*cstr != '\0') ? (stringLength(cstr + 1) + 1) : 0;
}

constexpr std::uint32_t sumSHL(std::uint32_t h, std::uint32_t shift) { return h + (h << shift); }
constexpr std::uint32_t sumSHR(std::uint32_t h, std::uint32_t shift) { return h + (h >> shift); }
constexpr std::uint32_t xorSHR(std::uint32_t h, std::uint32_t shift) { return h ^ (h >> shift); }

constexpr std::uint32_t hashFinishImpl(std::uint32_t h)
{
	// h += (h <<  3)
	// h ^= (h >> 11)
	// h += (h << 15)
	return sumSHL(xorSHR(sumSHL(h, 3), 11), 15);
}

constexpr std::uint32_t hashStepImpl(std::uint32_t h, std::uint32_t c)
{
	// h += c
	// h += (h << 10)
	// h ^= (h >>  6)
	return xorSHR(sumSHL(h + c, 10), 6);
}

constexpr std::uint32_t hashImpl(const char* cstr, std::uint32_t length, std::uint32_t h)
{
	return (length != 0) ? hashImpl(cstr + 1, length - 1, hashStepImpl(h, *cstr)) : hashFinishImpl(h);
}

constexpr std::uint32_t hashCString(const char* cstr)
{
	return hashImpl(cstr, stringLength(cstr), 0);
}

#if LOG 
	#define LOG_OGL(func)	                     \
        /*Log::warn((std::to_string(hashCString(#func)) + "\n").c_str());*/ \
		Log::warn(#func); \
        func                                                         
#else
	#define LOG_OGL(func) func
#endif

namespace PhysicsEngine
{
	class Graphics
	{
		public:
			static void checkError();
			static void checkFrambufferError();
			static GLenum getTextureFormat(TextureFormat format);

			static void create(Camera* camera, 
							   GLuint* mainFBO, 
							   GLuint* colorTex, 
							   GLuint* depthTex, 
							   GLuint* geometryFBO, 
							   GLuint* positionTex, 
							   GLuint* normalTex, 
							   GLuint* ssaoFBO, 
							   GLuint* ssaoColorTex, 
							   GLuint* ssaoNoiseTex,
							   glm::vec3* ssaoSamples,
							   bool* created);

			static void destroy(Camera* camera,
								GLuint* mainFBO,
								GLuint* colorTex,
								GLuint* depthTex,
								GLuint* geometryFBO,
								GLuint* positionTex,
								GLuint* normalTex,
								GLuint* ssaoFBO,
								GLuint* ssaoColorTex,
								GLuint* ssaoNoiseTex,
								bool* created);

			static void create(Texture2D* texture, GLuint* tex, bool* created);
			static void destroy(Texture2D* texture, GLuint* tex, bool* created);
			static void readPixels(Texture2D* texture);
			static void apply(Texture2D* texture);

			static void create(Texture3D* texture, GLuint* tex, bool* created);
			static void destroy(Texture3D* texture, GLuint* tex, bool* created);
			static void readPixels(Texture3D* texture);
			static void apply(Texture3D* texture);

			static void create(Cubemap* cubemap, GLuint* tex, bool* created);
			static void destroy(Cubemap* cubemap, GLuint* tex, bool* created);
			static void readPixels(Cubemap* cubemap);
			static void apply(Cubemap* cubemap);

			static void create(Mesh* mesh, GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2, bool* created);
			static void destroy(Mesh* mesh, GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2, bool* created);
			static void apply(Mesh* mesh);

			static void render(World* world, Material* material, int variant, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query);
			static void render(World* world, Shader* shader, int variant, Texture2D* texture, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query);
			static void render(World* world, Shader* shader, int variant, glm::mat4 model, GLuint vao, GLenum mode, int numVertices, GraphicsQuery* query);
			static void renderText(World* world, Camera* camera, Font* font, std::string text, float x, float y, float scale, glm::vec3 color);
			static void render(World* world, RenderObject renderObject, GraphicsQuery* query);
	};
}

#endif