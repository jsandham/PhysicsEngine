#include <iostream>
#include <objbase.h>
#include <stdio.h>

#include "../../include/core/Guid.h"

using namespace PhysicsEngine;

const Guid Guid::INVALID = Guid("00000000-0000-0000-0000-000000000000");

Guid::Guid()
{
	for(int i = 0; i < 16; i++){
		this->bytes[i] = '\0';
	}
}

Guid::Guid(const Guid& guid)
{
	for(int i = 0; i < 16; i++){
		this->bytes[i] = guid.bytes[i];
	}
}

Guid::Guid(const std::string &str)
{
	char c1 = '\0';
	char c2 = '\0';
	bool firstCharFound = false;
	int byteIndex = 0;
	for(int i = 0; i < str.length(); i++){
		if(str[i] == '-'){
			continue;
		}

		// return if char is invalid
		if(str[i] <= 47 || (str[i] >= 58 && str[i] <= 64) || (str[i] >= 71 && str[i] <= 96) || str[i] >= 103){
			for(int j = 0; j < 16; j++){
				bytes[j] = 0;
			}

			return;
		}

		if(!firstCharFound){
			c1 = str[i];
			firstCharFound = true;
		}
		else{
			c2 = str[i];
			firstCharFound = false;

			unsigned char b1, b2;

			if(c1 > 47 && c1 < 58){
				b1 = c1 - 48;
			}
			else if(c1 > 64 && c1 < 71){
				b1 = c1 - 55;
			}
			else if(c1 > 96 && c1 < 103){
				b1 = c1 - 87;
			}
			else{
				b1 = 0;
			}

			if(c2 > 47 && c2 < 58){
				b2 = c2 - 48;
			}
			else if(c2 > 64 && c2 < 71){
				b2 = c2 - 55;
			}
			else if(c2 > 96 && c2 < 103){
				b2 = c2 - 87;
			}
			else{
				b2 = 0;
			}

			bytes[byteIndex++] = 16 * b1 + b2;
		}

	}

	// check if guid is valid
	if(byteIndex < 16){
		for(int j = 0; j < 16; j++){
			bytes[j] = 0;
		}
	}
}

Guid::Guid(const std::vector<unsigned char> &bytes)
{
	for(int i = 0; i < 16; i++){
		this->bytes[i] = bytes[i];
	}
}

Guid::Guid(const unsigned char* bytes)
{
	for(int i = 0; i < 16; i++){
		this->bytes[i] = bytes[i];
	}
}

Guid::~Guid()
{

}

Guid& Guid::operator=(const Guid& guid)
{
	if(this != &guid)
	{
		for(int i = 0; i < 16; i++){
			bytes[i] = guid.bytes[i];
		}
	}

	return *this;
}

bool Guid::operator==(const Guid& guid) const
{
	for(int i = 0; i < 16; i++){
		if(bytes[i] != guid.bytes[i]){
			return false;
		}
	}

	return true;
}

bool Guid::operator!=(const Guid& guid) const
{
	for(int i = 0; i < 16; i++){
		if(bytes[i] != guid.bytes[i]){
			return true;
		}
	}

	return false;
}

bool Guid::operator<(const Guid& guid) const
{
	return ( memcmp( this, &guid, sizeof(Guid) ) > 0 ? true : false );

	//return (*this != guid);  // why doesnt this work??
}

bool Guid::isEmpty() const
{
	for(int i = 0; i < 16; i++){
		if(bytes[i] != '\0'){
			return false;
		}
	}

	return true;
}

std::string Guid::toString() const
{
	char one[10], two[6], three[6], four[6], five[14];

	_snprintf(one, 10, "%02x%02x%02x%02x", bytes[0], bytes[1], bytes[2], bytes[3]);

	_snprintf(two, 6, "%02x%02x", bytes[4], bytes[5]);

	_snprintf(three, 6, "%02x%02x", bytes[6], bytes[7]);

	_snprintf(four, 6, "%02x%02x", bytes[8], bytes[9]);

	_snprintf(five, 14, "%02x%02x%02x%02x%02x%02x", bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);

	const std::string sep("-");

	std::string out(one);

	out += sep + two;
	out += sep + three;
	out += sep + four;
	out += sep + five;

	return out;
}

Guid Guid::newGuid()
{
	GUID newId;

	CoCreateGuid(&newId);

	std::vector<unsigned char> bytes(16);

	bytes[0] = (unsigned char)((newId.Data1 >> 24) & 0xFF);
	bytes[1] = (unsigned char)((newId.Data1 >> 16) & 0xFF);
	bytes[2] = (unsigned char)((newId.Data1 >> 8) & 0xFF);
	bytes[3] = (unsigned char)((newId.Data1) & 0xff);

	bytes[4] = (unsigned char)((newId.Data2 >> 8) & 0xFF);
	bytes[5] = (unsigned char)((newId.Data2) & 0xff);

	bytes[6] = (unsigned char)((newId.Data3 >> 8) & 0xFF);
	bytes[7] = (unsigned char)((newId.Data3) & 0xFF);

	bytes[8] = (unsigned char)newId.Data4[0];
	bytes[9] = (unsigned char)newId.Data4[1];
	bytes[10] = (unsigned char)newId.Data4[2];
	bytes[11] = (unsigned char)newId.Data4[3];
	bytes[12] = (unsigned char)newId.Data4[4];
	bytes[13] = (unsigned char)newId.Data4[5];
	bytes[14] = (unsigned char)newId.Data4[6];
	bytes[15] = (unsigned char)newId.Data4[7];

	return bytes;
}

std::ostream& operator<<(std::ostream& os, const Guid& id)
{
	os << id.toString();
	return os;
}