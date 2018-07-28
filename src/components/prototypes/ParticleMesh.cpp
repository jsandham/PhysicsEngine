//#include "ParticleMesh.h"
//
//using namespace PhysicsEngine;
//
//ParticleMesh::ParticleMesh()
//{
//	vbo[0] = new VertexBufferObject(GL_DYNAMIC_DRAW);
//	vbo[1] = new VertexBufferObject(GL_DYNAMIC_DRAW);
//
//	vao = new VertexArrayObject(GL_POINTS);
//
//	vao->bind();
//	vbo[0]->bind();
//	vbo[0]->setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
//
//	vbo[1]->bind();
//	vbo[1]->setLayout(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0); 
//	vao->unbind();
//}
//
//ParticleMesh::ParticleMesh(Entity *entity)
//{
//	this->entity = entity;
//
//	vbo[0] = new VertexBufferObject(GL_DYNAMIC_DRAW);
//	vbo[1] = new VertexBufferObject(GL_DYNAMIC_DRAW);
//
//	vao = new VertexArrayObject(GL_POINTS);
//
//	vao->bind();
//	vbo[0]->bind();
//	vbo[0]->setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
//
//	vbo[1]->bind();
//	vbo[1]->setLayout(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);
//	vao->unbind();
//}
//
//ParticleMesh::~ParticleMesh()
//{
//	for (int i = 0; i < 2; i++){
//		delete vbo[i];
//	}
//	delete vao;
//}
//
//Material* ParticleMesh::getMaterial()
//{
//	return material;
//}
//
//std::vector<float>& ParticleMesh::getPoints()
//{
//	return points;
//}
//
//std::vector<float>& ParticleMesh::getTexCoords()
//{
//	return texCoords;
//}
//
//void ParticleMesh::setMaterial(Material *material)
//{
//	this->material = material;
//}
//
//void ParticleMesh::setPoints(std::vector<float> &points)
//{
//	this->points = points;
//
//	vao->bind();
//	vbo[0]->bind();
//	vbo[0]->setData(points);
//}
//
//void ParticleMesh::setTexCoords(std::vector<float> &texCoords)
//{
//	this->texCoords = texCoords;
//
//	vao->bind();
//	vbo[1]->bind();
//	vbo[1]->setData(texCoords);
//}
//
//void ParticleMesh::draw()
//{
//	vbo[0]->bind();
//	vbo[0]->setSubData(points);
//	vbo[0]->unbind();
//
//	vao->bind();
//	vao->draw((int)points.size());
//	vao->unbind();
//}