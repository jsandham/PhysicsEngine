#include "../include/PerformanceQueue.h"

using namespace PhysicsEditor;

PerformanceQueue::PerformanceQueue()
{
	setNumberOfSamples(10);
}

PerformanceQueue::~PerformanceQueue()
{

}

void PerformanceQueue::setNumberOfSamples(int numberOfSamples)
{
	this->numberOfSamples = numberOfSamples;
	index = 0;
	queue.resize(numberOfSamples, 0.0f);
	data.resize(numberOfSamples, 0.0f);

}

void PerformanceQueue::addSample(float value)
{
	queue[index] = value;
	index = (index + 1) % numberOfSamples;
}

void PerformanceQueue::clear()
{
	index = 0;
	for (size_t i = 0; i < queue.size(); i++) {
		queue[i] = 0.0f;
		data[i] = 0.0f;
	}
}

std::vector<float> PerformanceQueue::getData()
{
	int j = 0;
	for (size_t i = index; i < queue.size(); i++) {
		data[j] = queue[i];
		j++;
	}

	for (size_t i = 0; i < index; i++) {
		data[j] = queue[i];
		j++;
	}

	return data;
}