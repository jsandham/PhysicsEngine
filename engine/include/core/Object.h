// #ifndef OBJECT_H__
// #define OBJECT_H__

// #include "Guid.h"
// #include "Id.h"

// namespace PhysicsEngine
// {
// enum class HideFlag
// {
//     None = 0,
//     DontSave = 1
// };

// class World;

// class Object
// {
//   private:
//     Guid mGuid;
//     Id mId;

//   protected:
//     World *mWorld;

//   public:
//     HideFlag mHide;

//   public:
//     Object(World *world, const Id &id);
//     Object(World *world, const Guid &guid, const Id &id);
//     virtual ~Object() = 0;

//     virtual void serialize(YAML::Node &out) const;
//     virtual void deserialize(const YAML::Node &in);

//     virtual int getType() const = 0;
//     virtual std::string getObjectName() const = 0;

//     Guid getGuid() const;
//     Id getId() const;
// };

// } // namespace PhysicsEngine

// namespace YAML
// {
// // HideFlag
// template <> struct convert<PhysicsEngine::HideFlag>
// {
//     static Node encode(const PhysicsEngine::HideFlag &rhs)
//     {
//         Node node;
//         node = static_cast<int>(rhs);
//         return node;
//     }

//     static bool decode(const Node &node, PhysicsEngine::HideFlag &rhs)
//     {
//         rhs = static_cast<PhysicsEngine::HideFlag>(node.as<int>());
//         return true;
//     }
// };
// } // namespace YAML

// #endif