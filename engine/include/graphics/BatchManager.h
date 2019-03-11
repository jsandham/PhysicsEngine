#ifndef __BATCH_H__
#define __BATCH_H__

#include <vector>
#include <map>

#include <GL/glew.h>

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

		void add(Mesh* mesh)
		{
			if(mesh->vertices.size() != mesh->normals.size() || 2*mesh->vertices.size()/3 != mesh->texCoords.size()){
				std::cout << "Error: Cannot add mesh to batch" << std::endl;
				return;
			}

			unsigned int verticesOffset = 3 * currentNumOfVertices;
			unsigned int normalsOffset = 3 * currentNumOfVertices;
			unsigned int texCoordsOffset = 2 * currentNumOfVertices;
		
			glBindVertexArray(VAO);

			glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
			glBufferSubData( GL_ARRAY_BUFFER, verticesOffset * sizeof(float), mesh->vertices.size() * sizeof(float), &mesh->vertices[0] );

			glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
			glBufferSubData( GL_ARRAY_BUFFER, normalsOffset * sizeof(float), mesh->normals.size() * sizeof(float), &mesh->normals[0] );

			glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
			glBufferSubData( GL_ARRAY_BUFFER, texCoordsOffset * sizeof(float), mesh->texCoords.size() * sizeof(float), &mesh->texCoords[0] );

			glBindVertexArray(0);

			currentNumOfVertices += (unsigned int)mesh->vertices.size() / 3;

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

			void add(Material* material, Mesh* mesh);
			void render(World* world);
			void render(World* world, Material* material);
	};
}

#endif