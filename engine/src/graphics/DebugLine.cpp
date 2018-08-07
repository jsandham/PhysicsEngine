#include "../../include/graphics/DebugLine.h"

using namespace PhysicsEngine;

DebugLine::DebugLine(glm::vec3 start, glm::vec3 end)
{
	/*vertices.resize(6);
	vertices[0] = start.x;
	vertices[1] = start.y;
	vertices[2] = start.z;
	vertices[3] = end.x;
	vertices[4] = end.y;
	vertices[5] = end.z;

	vertexVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
	vertexVAO.generate();

	vertexVAO.setDrawMode(GL_LINES);

	vertexVAO.bind();
	vertexVBO.bind();
	vertexVBO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
	vertexVBO.setData(&vertices, vertices.size()*sizeof(float));
	vertexVBO.unbind();
	vertexVAO.unbind();*/
}

DebugLine::~DebugLine()
{

}

void DebugLine::draw()
{
	/*vertexVAO.bind();
	vertexVAO.draw((int)vertices.size());
	vertexVAO.unbind();*/
}