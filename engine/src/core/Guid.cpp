#include <iostream>
#include <objbase.h>
#include <stdio.h>

#include "../../include/core/Guid.h"

using namespace PhysicsEngine;

const Guid Guid::INVALID = Guid("00000000-0000-0000-0000-000000000000");

Guid::Guid()
{
    for (int i = 0; i < 16; i += 4)
    {
        this->bytes[i] = '\0';
        this->bytes[i + 1] = '\0';
        this->bytes[i + 2] = '\0';
        this->bytes[i + 3] = '\0';
    }
}

Guid::Guid(const Guid &guid)
{
    for (int i = 0; i < 16; i += 4)
    {
        this->bytes[i] = guid.bytes[i];
        this->bytes[i + 1] = guid.bytes[i + 1];
        this->bytes[i + 2] = guid.bytes[i + 2];
        this->bytes[i + 3] = guid.bytes[i + 3];
    }
}

Guid::Guid(const std::string &str)
{
    char c1 = '\0';
    char c2 = '\0';
    bool firstCharFound = false;
    int byteIndex = 0;
    for (size_t i = 0; i < str.length(); i++)
    {
        if (str[i] == '-')
        {
            continue;
        }

        // return if char is invalid
        if (str[i] <= 47 || (str[i] >= 58 && str[i] <= 64) || (str[i] >= 71 && str[i] <= 96) || str[i] >= 103)
        {
            for (int j = 0; j < 16; j++)
            {
                bytes[j] = 0;
            }

            return;
        }

        if (!firstCharFound)
        {
            c1 = str[i];
            firstCharFound = true;
        }
        else
        {
            c2 = str[i];
            firstCharFound = false;

            unsigned char b1, b2;

            if (c1 > 47 && c1 < 58)
            {
                b1 = c1 - 48;
            }
            else if (c1 > 64 && c1 < 71)
            {
                b1 = c1 - 55;
            }
            else if (c1 > 96 && c1 < 103)
            {
                b1 = c1 - 87;
            }
            else
            {
                b1 = 0;
            }

            if (c2 > 47 && c2 < 58)
            {
                b2 = c2 - 48;
            }
            else if (c2 > 64 && c2 < 71)
            {
                b2 = c2 - 55;
            }
            else if (c2 > 96 && c2 < 103)
            {
                b2 = c2 - 87;
            }
            else
            {
                b2 = 0;
            }

            bytes[byteIndex++] = 16 * b1 + b2;
        }
    }

    // check if guid is valid
    if (byteIndex < 16)
    {
        for (int j = 0; j < 16; j++)
        {
            bytes[j] = 0;
        }
    }
}

Guid::Guid(const std::vector<unsigned char> &bytes)
{
    for (int i = 0; i < 16; i += 4)
    {
        this->bytes[i] = bytes[i];
        this->bytes[i + 1] = bytes[i + 1];
        this->bytes[i + 2] = bytes[i + 2];
        this->bytes[i + 3] = bytes[i + 3];
    }
}

Guid::Guid(const unsigned char *bytes)
{
    for (int i = 0; i < 16; i += 4)
    {
        this->bytes[i] = bytes[i];
        this->bytes[i + 1] = bytes[i + 1];
        this->bytes[i + 2] = bytes[i + 2];
        this->bytes[i + 3] = bytes[i + 3];
    }
}

Guid::~Guid()
{
}

Guid &Guid::operator=(const Guid &guid)
{
    if (this != &guid)
    {
        for (int i = 0; i < 16; i += 4)
        {
            bytes[i] = guid.bytes[i];
            bytes[i + 1] = guid.bytes[i + 1];
            bytes[i + 2] = guid.bytes[i + 2];
            bytes[i + 3] = guid.bytes[i + 3];
        }
    }

    return *this;
}

bool Guid::operator==(const Guid &guid) const
{
    for (int i = 0; i < 16; i++)
    {
        if (bytes[i] != guid.bytes[i])
        {
            return false;
        }
    }

    return true;
}

bool Guid::operator!=(const Guid &guid) const
{
    for (int i = 0; i < 16; i++)
    {
        if (bytes[i] != guid.bytes[i])
        {
            return true;
        }
    }

    return false;
}

bool Guid::operator<(const Guid &guid) const
{
    return (memcmp(this, &guid, sizeof(Guid)) > 0 ? true : false);
}

bool Guid::isEmpty() const
{
    for (int i = 0; i < 16; i++)
    {
        if (bytes[i] != '\0')
        {
            return false;
        }
    }

    return true;
}

bool Guid::isValid() const
{
    return *this != Guid::INVALID;
}

bool Guid::isInvalid() const
{
    return *this == Guid::INVALID;
}

std::string Guid::toString() const
{
    char buffer[37];

    const char *format = "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x";

    _snprintf(buffer, 37, format, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
              bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);

    return std::string(buffer);
}

Guid Guid::newGuid()
{
    GUID newId;

    CoCreateGuid(&newId);

    const unsigned char bytes[16] = {(unsigned char)((newId.Data1 >> 24) & 0xFF),
                                     (unsigned char)((newId.Data1 >> 16) & 0xFF),
                                     (unsigned char)((newId.Data1 >> 8) & 0xFF),
                                     (unsigned char)((newId.Data1) & 0xff),

                                     (unsigned char)((newId.Data2 >> 8) & 0xFF),
                                     (unsigned char)((newId.Data2) & 0xff),

                                     (unsigned char)((newId.Data3 >> 8) & 0xFF),
                                     (unsigned char)((newId.Data3) & 0xFF),

                                     (unsigned char)newId.Data4[0],
                                     (unsigned char)newId.Data4[1],
                                     (unsigned char)newId.Data4[2],
                                     (unsigned char)newId.Data4[3],
                                     (unsigned char)newId.Data4[4],
                                     (unsigned char)newId.Data4[5],
                                     (unsigned char)newId.Data4[6],
                                     (unsigned char)newId.Data4[7]};

    return Guid(&bytes[0]);
}

std::ostream &operator<<(std::ostream &os, const Guid &id)
{
    os << id.toString();
    return os;
}