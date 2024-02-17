#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/graphics/platform/opengl/OpenGLGraphicsQuery.h"

#include <assert.h>

using namespace PhysicsEngine;

OpenGLOcclusionQuery::OpenGLOcclusionQuery()
{
}

OpenGLOcclusionQuery::~OpenGLOcclusionQuery()
{
    GLsizei queryCount = (GLsizei)mQueryIds.size();
    CHECK_ERROR(glDeleteQueries(queryCount, mQueryIds.data()));
}

void OpenGLOcclusionQuery::increaseQueryCount(size_t count)
{
    size_t oldCount = mQueryIds.size();
    GLsizei queryCount = GLsizei(count - oldCount);
    if (queryCount > 0)
    {
        mQueryIds.resize(count);
        CHECK_ERROR(glGenQueries(queryCount, &mQueryIds[oldCount]));
    }
}

void OpenGLOcclusionQuery::beginQuery(size_t queryIndex)
{
    assert(queryIndex < mQueryIds.size());

    CHECK_ERROR(glBeginQuery(GL_SAMPLES_PASSED, mQueryIds[queryIndex]));
}

void OpenGLOcclusionQuery::endQuery(size_t queryIndex)
{
    assert(queryIndex < mQueryIds.size());

    CHECK_ERROR(glEndQuery(GL_SAMPLES_PASSED));
}

bool OpenGLOcclusionQuery::isVisible(size_t queryIndex)
{
    assert(queryIndex < mQueryIds.size());

    GLuint result;
    CHECK_ERROR(glGetQueryObjectuiv(mQueryIds[queryIndex], GL_QUERY_RESULT, &result));

    return result != 0;
}

bool OpenGLOcclusionQuery::isVisibleNoWait(size_t queryIndex)
{
    assert(queryIndex < mQueryIds.size());

    GLuint result;
    CHECK_ERROR(glGetQueryObjectuiv(mQueryIds[queryIndex], GL_QUERY_RESULT_AVAILABLE, &result));

    if (result != 0)
    {
        CHECK_ERROR(glGetQueryObjectuiv(mQueryIds[queryIndex], GL_QUERY_RESULT, &result));
    }

    return result != 0;
}
