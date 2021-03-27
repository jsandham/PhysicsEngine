#ifndef __RIGIDBODY_DRAWER_H__
#define __RIGIDBODY_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class RigidbodyDrawer : public InspectorDrawer
{
  public:
    RigidbodyDrawer();
    ~RigidbodyDrawer();

    void render(Clipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif