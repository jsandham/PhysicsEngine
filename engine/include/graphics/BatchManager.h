#ifndef __BATCHMANAGER_H__
#define __BATCHMANAGER_H__

#include <vector>
#include <map>

#include <GL/glew.h>

#include "GraphicsQuery.h"

#include "../glm/glm.hpp"

#include "../core/World.h"
#include "../core/Mesh.h"
#include "../core/Material.h"
#include "../core/Shader.h"
#include "../core/Texture2D.h"

namespace PhysicsEngine
{
	struct Batch
	{
		unsigned int maxNumOfVertices;
		unsigned int maxNumOfMeshes;
		unsigned int currentNumOfVertices;
		unsigned int currentNumOfMeshes;
		GLuint VAO;
		GLuint vertexVBO;
		GLuint normalVBO;
		GLuint texCoordVBO;
		Guid materialId;

		Batch()
		{
			maxNumOfVertices = 0;
			maxNumOfMeshes = 0;
			currentNumOfVertices = 0;
			currentNumOfMeshes = 0;
			VAO = 0;
			vertexVBO = 0;
			normalVBO = 0;
			texCoordVBO = 0;
			materialId = Guid::INVALID;
		}

		bool hasEnoughRoom(unsigned int numVertices)
		{
			if((currentNumOfVertices + numVertices) > maxNumOfVertices){
	 			return false; 
			}

			return true;
		}

		void add(Mesh* mesh, glm::mat4 model)
		{
			if(mesh->getVertices().size() != mesh->getNormals().size() || 2*mesh->getVertices().size()/3 != mesh->getTexCoords().size()){
				std::cout << "Error: Cannot add mesh to batch" << std::endl;
				return;
			}

			std::vector<float> worldSpaceVertices;
			worldSpaceVertices.resize(mesh->getVertices().size());

			for(size_t i = 0; i < mesh->getVertices().size() / 3; i++){
				glm::vec4 vertex = model * glm::vec4(mesh->getVertices()[3*i], mesh->getVertices()[3*i + 1], mesh->getVertices()[3*i + 2], 1.0f);
				worldSpaceVertices[3*i] = vertex.x;
				worldSpaceVertices[3*i + 1] = vertex.y;
				worldSpaceVertices[3*i + 2] = vertex.z;
			}

			unsigned int verticesOffset = 3 * currentNumOfVertices;
			unsigned int normalsOffset = 3 * currentNumOfVertices;
			unsigned int texCoordsOffset = 2 * currentNumOfVertices;
		
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
			glBufferSubData( GL_ARRAY_BUFFER, verticesOffset * sizeof(float), worldSpaceVertices.size() * sizeof(float), &worldSpaceVertices[0] );

			glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
			glBufferSubData( GL_ARRAY_BUFFER, normalsOffset * sizeof(float), mesh->getNormals().size() * sizeof(float), &mesh->getNormals()[0] );

			glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
			glBufferSubData( GL_ARRAY_BUFFER, texCoordsOffset * sizeof(float), mesh->getTexCoords().size() * sizeof(float), &mesh->getTexCoords()[0] );

			glBindVertexArray(0);

			currentNumOfVertices += (unsigned int)worldSpaceVertices.size() / 3;

			GLenum error;
			while ((error = glGetError()) != GL_NO_ERROR){
				std::cout << "Error: Renderer failed with error code: " << error << " during add" << std::endl;;
			}
		}

		void generate()
		{
			maxNumOfVertices = 3 * (maxNumOfVertices / 3);  // ensure multiple of 3

			glGenVertexArrays(1, &VAO);
			glBindVertexArray(VAO);

			glGenBuffers(1, &vertexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
			glBufferData(GL_ARRAY_BUFFER, 3 * maxNumOfVertices*sizeof(float), NULL, GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glGenBuffers(1, &normalVBO);
			glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
			glBufferData(GL_ARRAY_BUFFER, 3 * maxNumOfVertices*sizeof(float), NULL, GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glGenBuffers(1, &texCoordVBO);
			glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
			glBufferData(GL_ARRAY_BUFFER, 2 * maxNumOfVertices*sizeof(float), NULL, GL_STATIC_DRAW);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

			glBindVertexArray(0);

			GLenum error;
			while ((error = glGetError()) != GL_NO_ERROR){
				std::cout << "Error: Renderer failed with error code: " << error << " during generate" << std::endl;;
			}
		}

		void destroy()
		{
			if( vertexVBO != 0 ) {
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDeleteBuffers( 1, &vertexVBO );
				vertexVBO = 0;
			}
			if( normalVBO != 0 ) {
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDeleteBuffers( 1, &normalVBO );
				normalVBO = 0;
			}
			if( texCoordVBO != 0 ) {
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDeleteBuffers( 1, &texCoordVBO );
				texCoordVBO = 0;
			}

			if( VAO != 0 ) {
				glBindVertexArray( 0 );
				glDeleteVertexArrays( 1, &VAO );
				VAO = 0;
			}
		}
	};


	class BatchManager
	{
		private:
			unsigned int maxNumOfVerticesPerBatch;
			unsigned int maxNumOfMeshesPerBatch;
			std::map<Guid, std::vector<Batch>> materialIdToBatchesMap;

		public:
			BatchManager();
			BatchManager(unsigned int maxNumOfVerticesPerBatch, unsigned int maxNumOfMeshesPerBatch);
			~BatchManager();

			void add(Material* material, Mesh* mesh, glm::mat4 model);

			void render(World* world, int variant, GraphicsQuery* query);
			void render(World* world, Material* material, int variant, GraphicsQuery* query); // do I even need this now?? I think I was using this for rendering debug info.
			void render(World* world, Shader* shader, int variant, GraphicsQuery* query);
	};
}

#endif