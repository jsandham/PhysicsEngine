#include "../../include/core/DebugWindow.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

DebugWindow::DebugWindow(float x, float y, float width, float height)
{
	this->x = fmin(fmax(x, 0.0f), 1.0f);
	this->y = fmin(fmax(y, 0.0f), 1.0f);
	this->width = fmin(fmax(width, 0.0f), 1.0f);
	this->height = fmin(fmax(height, 0.0f), 1.0f);

	vertices.resize(18);

	float x_ndc = 2.0f * x - 1.0f; 
	float y_ndc = 1.0f - 2.0f * y; 

	float width_ndc = 2.0f * width;
	float height_ndc = 2.0f * height;

	vertices[0] = x_ndc; 
	vertices[1] = y_ndc;
	vertices[2] = 0.0f;

	vertices[3] = x_ndc; 
	vertices[4] = y_ndc - height_ndc; 
	vertices[5] = 0.0f; 

	vertices[6] = x_ndc + width_ndc; 
	vertices[7] = y_ndc;
	vertices[8] = 0.0f;  

	vertices[9] = x_ndc + width_ndc; 
	vertices[10] = y_ndc;  
	vertices[11] = 0.0f; 

	vertices[12] = x_ndc; 
	vertices[13] = y_ndc - height_ndc; 
	vertices[14] = 0.0f;  

	vertices[15] = x_ndc + width_ndc; 
	vertices[16] = y_ndc - height_ndc; 
	vertices[17] = 0.0f; 

	texCoords.resize(12);

	texCoords[0] = 0.0f;
	texCoords[1] = 1.0f;

	texCoords[2] = 0.0f;
	texCoords[3] = 0.0f;

	texCoords[4] = 1.0f;
	texCoords[5] = 1.0f;

	texCoords[6] = 1.0f;
	texCoords[7] = 1.0f;

	texCoords[8] = 0.0f;
	texCoords[9] = 0.0f;

	texCoords[10] = 1.0f;
	texCoords[11] = 0.0f;
}

DebugWindow::~DebugWindow()
{
	
}