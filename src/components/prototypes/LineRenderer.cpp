#include "LineRenderer.h"

using namespace PhysicsEngine;

LineRenderer::LineRenderer()
{
	
}

LineRenderer::~LineRenderer()
{

}

void LineRenderer::initLineData()
{
	vertexVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
	lineVAO.generate();

	lineVAO.setDrawMode(GL_LINES);
	lineVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	lineVAO.bind();
	vertexVBO.bind();
	/*vertexVBO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);*/
	vertexVBO.setData(&vertices[0], vertices.size()*sizeof(float));
}

void LineRenderer::updateLineData()
{
	lineVAO.bind();
	vertexVBO.bind();
	vertexVBO.setData(&vertices[0], vertices.size()*sizeof(float));
}

void LineRenderer::draw()
{
	lineVAO.bind();
	lineVAO.draw((int)vertices.size());
	lineVAO.unbind();
}

void LineRenderer::setQueued(bool flag)
{
	queued = flag;
}

void LineRenderer::setMaterialFilter(int filter)
{
	this->matFilter = filter;
}

void LineRenderer::setLineWidth(float width)
{
	this->lineWidth = width;
}

void LineRenderer::setVertices(std::vector<float> vertices)
{
	this->vertices = vertices;
}

bool LineRenderer::isQueued()
{
	return queued == true;
}

int LineRenderer::getMaterialFilter()
{
	return matFilter;
}

float LineRenderer::getLineWidth()
{
	return lineWidth;
}

std::vector<float> LineRenderer::getVertices()
{
	return vertices;
}