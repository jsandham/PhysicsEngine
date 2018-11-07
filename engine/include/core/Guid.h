#ifndef __GUID_H__
#define __GUID_H__

#include <string>
#include <vector>

namespace PhysicsEngine
{
	// see https://github.com/graeme-hill/crossguid
	class Guid
	{
		public:
			unsigned char bytes[16];

		public:
			Guid();
			Guid(const std::vector<unsigned char> &bytes);
			Guid(const unsigned char* bytes);
			Guid(const Guid& guid);
			Guid(const std::string &str);
			~Guid();

			Guid& operator=(const Guid& guid);
			bool operator==(const Guid& guid) const;
			bool operator!=(const Guid& guid) const;
			bool operator<(const Guid& guid) const;
			
			bool isEmpty() const;
			std::string toString() const;

			static Guid newGuid();

			static const Guid INVALID;
	};
}

#endif