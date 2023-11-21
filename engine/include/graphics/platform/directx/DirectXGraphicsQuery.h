#ifndef DIRECTX_GRAPHICS_QUERY_H__
#define DIRECTX_GRAPHICS_QUERY_H__

#include "../../GraphicsQuery.h"

namespace PhysicsEngine
{
class DirectXOcclusionQuery : public OcclusionQuery
{
  public:
    DirectXOcclusionQuery();
    ~DirectXOcclusionQuery();

    void increaseQueryCount(size_t count) override;

    void beginQuery(size_t queryIndex) override;
    void endQuery(size_t queryIndex) override;

    bool isVisible(size_t queryIndex) override;
    bool isVisibleNoWait(size_t queryIndex) override;
};

class DirectXTimingQuery : public TimingQuery
{
};
} // namespace PhysicsEngine

#endif