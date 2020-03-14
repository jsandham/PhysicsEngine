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

    	// int getIndex(Guid meshId);
    	int getStartIndex(Guid meshId);
    	Sphere getBoundingSphere(Guid meshId);
	}MeshBuffer;


	class Graphics
	{
		public:
			static void checkError();
			static void checkFrambufferError();
			static GLenum getTextureFormat(TextureFormat format);

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