#ifndef ID_H__
#define ID_H__

namespace PhysicsEngine
{
    class Id
    {
    private:
        int mId;

    public:
        Id();
        Id(int id);
        ~Id();

        Id &operator=(const Id &id);
        bool operator==(const Id &id) const;
        bool operator!=(const Id &id) const;
        bool operator<(const Id &id) const;

        bool isValid() const;
        bool isInvalid() const;

        static Id newId();
        static const Id INVALID;
    };
}

#endif