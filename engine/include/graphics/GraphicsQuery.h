#ifndef GRAPHICSQUERY_H__
#define GRAPHICSQUERY_H__

namespace PhysicsEngine
{
struct OcclusionQuery
{
    unsigned int samplesDrawn;

    //void begin();
    //void end();
};

struct TimingQuery
{
    unsigned int mNumInstancedDrawCalls;
    unsigned int mNumDrawCalls;
    unsigned int mVerts;
    unsigned int mTris;
    unsigned int mLines;
    unsigned int mPoints;

    unsigned int mQueryBack;
    unsigned int mQueryFront;
    unsigned int mQueryId[2];
    float mTotalElapsedTime;

    TimingQuery()
    {
        mNumInstancedDrawCalls = 0;
        mNumDrawCalls = 0;
        mVerts = 0;
        mTris = 0;
        mLines = 0;
        mPoints = 0;

        mQueryBack = 0;
        mQueryFront = 0;
        mQueryId[0] = 0;
        mQueryId[1] = 0;
        mTotalElapsedTime = 0.0f;
    }
};
} // namespace PhysicsEngine

#endif