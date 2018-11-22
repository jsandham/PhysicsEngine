#include <iostream>
#include <math.h>

#include "../../include/core/PerformanceGraph.h"

using namespace PhysicsEngine;

PerformanceGraph::PerformanceGraph(float x, float y, float width, float height, float rangeMin, float rangeMax, int numberOfSamples)
{
	this->x = fmin(fmax(x, 0.0f), 1.0f);
	this->y = fmin(fmax(y, 0.0f), 1.0f);
	this->width = fmin(fmax(width, 0.0f), 1.0f);
	this->height = fmin(fmax(height, 0.0f), 1.0f);
	this->rangeMin = rangeMin;
	this->rangeMax = rangeMax;
	this->currentSample = 0.0f;
	this->numberOfSamples = numberOfSamples;

	vertices.resize(18*numberOfSamples - 18);
}

PerformanceGraph::~PerformanceGraph()
{

}

void PerformanceGraph::add(float sample)
{
	float oldSample = currentSample;
	currentSample = fmin(fmax(sample, rangeMin), rangeMax);

	float dx = width / (numberOfSamples - 1);
	for(int i = 0; i < numberOfSamples - 2; i++){
		vertices[18*i] = vertices[18*(i+1)] - dx;
		vertices[18*i + 1] = vertices[18*(i+1) + 1];
		vertices[18*i + 2] = vertices[18*(i+1) + 2];

		vertices[18*i + 3] = vertices[18*(i+1) + 3] - dx;
		vertices[18*i + 4] = vertices[18*(i+1) + 4];
		vertices[18*i + 5] = vertices[18*(i+1) + 5];

		vertices[18*i + 6] = vertices[18*(i+1) + 6] - dx;
		vertices[18*i + 7] = vertices[18*(i+1) + 7];
		vertices[18*i + 8] = vertices[18*(i+1) + 8];

		vertices[18*i + 9] = vertices[18*(i+1) + 9] - dx;
		vertices[18*i + 10] = vertices[18*(i+1) + 10];
		vertices[18*i + 11] = vertices[18*(i+1) + 11];

		vertices[18*i + 12] = vertices[18*(i+1) + 12] - dx;
		vertices[18*i + 13] = vertices[18*(i+1) + 13];
		vertices[18*i + 14] = vertices[18*(i+1) + 14];

		vertices[18*i + 15] = vertices[18*(i+1) + 15] - dx;
		vertices[18*i + 16] = vertices[18*(i+1) + 16];
		vertices[18*i + 17] = vertices[18*(i+1) + 17];
	}

	float dz1 = 1.0f - (currentSample - rangeMin) / (rangeMax - rangeMin);
	float dz2 = 1.0f - (oldSample - rangeMin) / (rangeMax - rangeMin);

	float x_ndc = 2.0f * x - 1.0f;
	float y0_ndc = 1.0f - 2.0f * (y + height);
	float y1_ndc = 1.0f - 2.0f * (y + height * dz1);
	float y2_ndc = 1.0f - 2.0f * (y + height * dz2);

	vertices[18*(numberOfSamples - 2)] = x_ndc + dx * (numberOfSamples - 2);
	vertices[18*(numberOfSamples - 2) + 1] = y2_ndc;
	vertices[18*(numberOfSamples - 2) + 2] = 0.0f;

	vertices[18*(numberOfSamples - 2) + 3] = x_ndc + dx * (numberOfSamples - 2);
	vertices[18*(numberOfSamples - 2) + 4] = y0_ndc;
	vertices[18*(numberOfSamples - 2) + 5] = 0.0f;

	vertices[18*(numberOfSamples - 2) + 6] = x_ndc + dx * (numberOfSamples - 1);
	vertices[18*(numberOfSamples - 2) + 7] = y0_ndc;
	vertices[18*(numberOfSamples - 2) + 8] = 0.0f;

	vertices[18*(numberOfSamples - 2) + 9] = x_ndc + dx * (numberOfSamples - 2); 
	vertices[18*(numberOfSamples - 2) + 10] = y2_ndc;
	vertices[18*(numberOfSamples - 2) + 11] = 0.0f;

	vertices[18*(numberOfSamples - 2) + 12] = x_ndc + dx * (numberOfSamples - 1);
	vertices[18*(numberOfSamples - 2) + 13] = y0_ndc;
	vertices[18*(numberOfSamples - 2) + 14] = 0.0f;

	vertices[18*(numberOfSamples - 2) + 15] = x_ndc + dx * (numberOfSamples - 1);
	vertices[18*(numberOfSamples - 2) + 16] = y1_ndc;
	vertices[18*(numberOfSamples - 2) + 17] = 0.0f;
}