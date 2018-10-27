#ifndef __GUID_H__
#define __GUID_H__

#include <string>
#include <vector>

// see https://github.com/graeme-hill/crossguid
class Guid
{
	private:
		std::vector<unsigned char> bytes;

	public:
		Guid();
		Guid(const std::vector<unsigned char> &bytes);
		Guid(const unsigned char* bytes);
		Guid(const Guid& guid);
		Guid(const std::string &str);
		~Guid();

		Guid& operator=(const Guid& guid);
		bool operator==(const Guid& guid);
		bool operator!=(const Guid& guid);
		
		std::string toString() const;

		static Guid newGuid();
};


#endif