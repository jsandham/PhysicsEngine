#ifndef __TRANSFORM_DRAWER_H__
#define __TRANSFORM_DRAWER_H__

#define GLM_FORCE_RADIANS

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class TransformDrawer : public InspectorDrawer
{
  public:
    TransformDrawer();
    ~TransformDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif