#ifndef __RIGIDBODY_DRAWER_H__
#define __RIGIDBODY_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class RigidbodyDrawer : public InspectorDrawer
{
  public:
    RigidbodyDrawer();
    ~RigidbodyDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif