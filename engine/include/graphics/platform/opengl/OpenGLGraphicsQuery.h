#ifndef OPENGL_GRAPHICS_QUERY_H__
#define OPENGL_GRAPHICS_QUERY_H__

#include "../../GraphicsQuery.h"

#include <GL/glew.h>
#include <vector>

namespace PhysicsEngine
{
	class OpenGLOcclusionQuery : public OcclusionQuery
	{
        private:
          std::vector<GLuint> mQueryIds;

        public:
          OpenGLOcclusionQuery();
          ~OpenGLOcclusionQuery();

          void increaseQueryCount(size_t count) override;

          void beginQuery(size_t queryIndex) override;
          void endQuery(size_t queryIndex) override;

          bool isVisible(size_t queryIndex) override;
          bool isVisibleNoWait(size_t queryIndex) override;
	};

	class OpenGLTimingQuery : public TimingQuery
    {
    };
} // namespace PhysicsEngine

#endif
