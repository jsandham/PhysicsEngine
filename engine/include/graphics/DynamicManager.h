#ifndef __DYNAMICMANAGER_H__
#define __DYNAMICMANAGER_H__

#include <vector>

namespace PhysicsEngine
{
	struct GraphicsMesh  //OpenGLMesh? InternalMesh? DynamicMesh?
	{
		GLuint VAO;
		GLuint vertexVBO;
		GLuint normalVBO;
		GLuint texCoordVBO;
		// Guid materialId;
		// Guid meshId;
	};

	class DynamicManager  //MeshManager?? Could just move functionality into Render class cirectly instead of having this class?
	{
		private:
			std::map<Guid, GraphicsMesh> meshIdToGraphicsMesh;

		public:
			DynamicManager();
			~DynamicManager();

			void add(Mesh* mesh);
			void render(World* world, glm::mat4 model, Mesh* mesh, Material* material);
	};
}

#endif