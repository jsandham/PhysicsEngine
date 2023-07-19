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
	mNumberOfSamples = numberOfSamples;
	mIndex = 0;
	mQueue.resize(numberOfSamples, 0.0f);
	mData.resize(numberOfSamples, 0.0f);
}

void PerformanceQueue::addSample(float value)
{
	mQueue[mIndex] = value;
	mIndex = (mIndex + 1) % mNumberOfSamples;
}

void PerformanceQueue::clear()
{
	mIndex = 0;
	for (size_t i = 0; i < mQueue.size(); i++)
	{
		mQueue[i] = 0.0f;
		mData[i] = 0.0f;
	}
}

std::vector<float> PerformanceQueue::getData()
{
	int j = 0;
	for (size_t i = mIndex; i < mQueue.size(); i++)
	{
		mData[j] = mQueue[i];
		j++;
	}

	for (size_t i = 0; i < mIndex; i++)
	{
		mData[j] = mQueue[i];
		j++;
	}

	return mData;
}