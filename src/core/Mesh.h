#ifndef __MESH_H__
#define __MESH_H__

#include<vector>

namespace PhysicsEngine
{
	class Mesh
	{
		private:
			std::vector<float> vertices;
			std::vector<float> normals;
			std::vector<float> texCoords;
			std::vector<float> colours;

		public:
			Mesh();
			~Mesh();

			std::vector<float>& getVertices();
			std::vector<float>& getNormals();
			std::vector<float>& getTexCoords();
			std::vector<float>& getColours();

			void setVertices(std::vector<float> &vertices);
			void setNormals(std::vector<float> &normals);
			void setTexCoords(std::vector<float> &texCoords);
			void setColours(std::vector<float>& colours);
	};
}

#endif