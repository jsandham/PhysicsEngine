#ifndef __UTIL_H__
#define __UTIL_H__

#include <vector>
#include <string>

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct BMPHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
		unsigned short reserved1;
		unsigned short reserved2;
		unsigned int bitmapOffset;
		unsigned int size;
		int width;
		int height;
		unsigned short planes;
		unsigned short bitsPerPixel;
		unsigned int compression;
		unsigned int sizeOfBitmap;
		int horizontalResolution;
		int verticalResolution;
		unsigned int colorsUsed;
		unsigned int colorsImportant;
	};
#pragma pack(pop)

	class Util
	{
		public:
			static bool writeToBMP(const std::string& filepath, std::vector<unsigned char>& data, int width, int height, int numChannels);
			static bool writeToBMP(const std::string& filepath, std::vector<float>& data, int width, int height, int numChannels);
	};
}

#endif