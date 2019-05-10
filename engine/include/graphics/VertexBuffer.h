// #ifndef __VERTEXBUFFER_H__
// #define __VERTEXBUFFER_H__

// #include <vector>

// #include "../glm/glm.hpp"

// namespace PhysicsEngine
// {
// 	struct VertexVNT
// 	{
// 		std::vector<float> vertices;
// 		std::Vector<float> normals;
// 		std::vector<float> texCoords;

// 		static void EnableVertexAttribArray()
// 	    {
// 	    	glEnableVertexAttribArray(0);
// 			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), vertices);



// 	        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SVertexP1N1), (const GLvoid*)offsetof(SVertexP1N1, m_position));
// 	        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SVertexP1N1), (const GLvoid*)offsetof(SVertexP1N1, m_normal));
// 	        glEnableVertexAttribArray(0);
// 	        glEnableVertexAttribArray(1);
// 	    }

// 	}

// 	template <class VertexType> 
// 	class VertexBuffer
// 	{
// 		private:
// 	    	GLuint vao;
// 	    	GLuint vertexVBO;
// 	    	GLuint normalVBO;
// 	    	GLuint texCoordVBO;

	    	
	    	
// 	    	std::vector<vertex_type> m_vertices;
// 	    	std::vector<GLuint> m_indices;

// 		public:
// 	    	typedef VertexType vertexType;

// 	    	VertexBuffer();
// 		    ~VertexBuffer();
// 	}
// }


// #endif