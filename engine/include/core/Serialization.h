#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include "Guid.h"

namespace PhysicsEngine
{
	template<class T>
	Guid ExtactAssetId(std::vector<char> data)
	{
		static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

		return Guid::INVALID;
	}

	template<class T>
	Guid ExtactComponentId(std::vector<char> data)
	{
		static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

		return Guid::INVALID;
	}

	template<class T>
	Guid ExtactSystemId(std::vector<char> data)
	{
		static_assert(IsSystem<T>::value == true, "'T' is not of type System");

		return Guid::INVALID;
	}
}

#endif