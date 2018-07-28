//#ifndef __PARTICLEMESH_H__
//#define __PARTICLEMESH_H__
//
//#include <string>
//#include <vector>
//
//#include "Component.h"
//
//#include "../Bounds.h"
//
//#include "../graphics/Material.h"
//#include "../graphics/VertexBufferObject.h"
//#include "../graphics/VertexArrayObject.h"
//
//namespace PhysicsEngine
//{
//	class ParticleMesh : public Component
//	{
//		private:
//			Material *material;
//			Bounds bounds;
//
//			VertexBufferObject* vbo[2];
//			VertexArrayObject* vao;
//
//			std::vector<float> points;
//			std::vector<float> texCoords;
//
//		public:
//			ParticleMesh();
//			ParticleMesh(Entity *entity);
//			~ParticleMesh();
//
//			Material* getMaterial();
//			std::vector<float>& getPoints();
//			std::vector<float>& getTexCoords();
//
//			void setMaterial(Material *material);
//			void setPoints(std::vector<float> &points);
//			void setTexCoords(std::vector<float> &texCoords);
//
//			void draw();
//	};
//}
//
//#endif