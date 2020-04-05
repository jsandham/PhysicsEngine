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