#ifndef __GIZMOS_H__
#define __GIZMOS_H__

#include "Color.h"
#include "Shader.h"
#include "Buffer.h"
#include "VertexArrayObject.h"

#include "../core/Mesh.h"
#include "../core/Octtree.h"

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class Gizmos
	{
		private:
			static Shader gizmoShader;

			static VertexArrayObject sphereVAO;
			static VertexArrayObject cubeVAO;
			static VertexArrayObject meshVAO;
			static VertexArrayObject treeVAO;

			static Buffer sphereVBO;
			static Buffer cubeVBO;
			static Buffer meshVBO;
			static Buffer treeVBO;

			static Mesh sphereMesh;
			static Mesh cubeMesh;

			static bool isInitialized;

		public:
			static glm::mat4 projection;
			static glm::mat4 view;

		public:
			static void init();
			static void drawSphere(glm::vec3 centre, float radius, Color color);
			static void drawCube(glm::vec3 centre, glm::vec3 size, Color color);
			static void drawMesh(Mesh* mesh, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, Color color);
			static void drawWireSphere(glm::vec3 centre, float radius, Color color);
			static void drawWireCube(glm::vec3 centre, glm::vec3 size, Color color);
			static void drawWireMesh(Mesh* mesh, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, Color color);
			static void drawOcttree(Octtree* tree, Color color);
	
		private:
			static void draw(VertexArrayObject vao, int numVertices, glm::mat4 model, Color color);
	};
}

#endif
