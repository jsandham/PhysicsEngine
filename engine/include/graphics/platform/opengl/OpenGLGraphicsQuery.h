#ifndef OPENGL_GRAPHICS_QUERY_H__
#define OPENGL_GRAPHICS_QUERY_H__

#include "../../GraphicsQuery.h"

namespace PhysicsEngine
{
	class OpenGLOcclusionQuery : public OcclusionQuery
	{
      private:
        unsigned int mQuery;

        public:
          OpenGLOcclusionQuery();
          ~OpenGLOcclusionQuery();

          void beginQuery() override;
          void endQuery() override;
	};

	class OpenGLTimingQuery : public TimingQuery
    {
    };
    } // namespace PhysicsEngine

#endif
