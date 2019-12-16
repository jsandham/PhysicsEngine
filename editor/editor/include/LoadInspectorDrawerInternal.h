#ifndef __LOAD_INSPECTOR_DRAWER_INTERNAL_H__
#define __LOAD_INSPECTOR_DRAWER_INTERNAL_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	InspectorDrawer* loadInternalInspectorComponentDrawer(int type);
	InspectorDrawer* loadInternalInspectorAssetDrawer(int type);
}

#endif