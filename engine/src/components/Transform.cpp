#include "../../include/components/Transform.h"

#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

#define GLM_ENABLE_EXPERIMENTAL

Transform::Transform(World* world) : Component(world)
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(World* world, Guid id) : Component(world, id)
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::~Transform()
{
}

void Transform::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["parentId"] = mParentId;
    out["position"] = mPosition;
    out["rotation"] = mRotation;
    out["scale"] = mScale;
}

void Transform::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mParentId = YAML::getValue<Guid>(in, "parentId");
    mPosition = YAML::getValue<glm::vec3>(in, "position");
    mRotation = YAML::getValue<glm::quat>(in, "rotation");
    mScale = YAML::getValue<glm::vec3>(in, "scale");
}

int Transform::getType() const
{
    return PhysicsEngine::TRANSFORM_TYPE;
}

std::string Transform::getObjectName() const
{
    return PhysicsEngine::TRANSFORM_NAME;
}

glm::mat4 Transform::getModelMatrix() const
{
    glm::mat4 modelMatrix = glm::translate(glm::mat4(), mPosition);
    modelMatrix *= glm::toMat4(mRotation);
    modelMatrix = glm::scale(modelMatrix, mScale);

    return modelMatrix;
}

glm::vec3 Transform::getForward() const
{
    // a transform with zero rotation has its blue axis pointing in the z direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 0, 1, 0)));
}

glm::vec3 Transform::getUp() const
{
    // a transform with zero rotation has its green axis pointing in the y direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 1, 0, 0)));
}

glm::vec3 Transform::getRight() const
{
    // a transform with zero rotation has its red axis pointing in the x direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(1, 0, 0, 0)));
}

bool Transform::decompose(const glm::mat4& model, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale)
{
	// From glm::decompose in matrix_decompose.inl
	using namespace glm;

	mat4 LocalMatrix(model);

	// Normalize the matrix.
	if (epsilonEqual(LocalMatrix[3][3], static_cast<float>(0), epsilon<float>()))
		return false;

	// First, isolate perspective.  This is the messiest.
	if (
		epsilonNotEqual(LocalMatrix[0][3], static_cast<float>(0), epsilon<float>()) ||
		epsilonNotEqual(LocalMatrix[1][3], static_cast<float>(0), epsilon<float>()) ||
		epsilonNotEqual(LocalMatrix[2][3], static_cast<float>(0), epsilon<float>()))
	{
		// Clear the perspective partition
		LocalMatrix[0][3] = LocalMatrix[1][3] = LocalMatrix[2][3] = static_cast<float>(0);
		LocalMatrix[3][3] = static_cast<float>(1);
	}

	// Next take care of translation (easy).
	translation = vec3(LocalMatrix[3]);
	LocalMatrix[3] = vec4(0, 0, 0, LocalMatrix[3].w);

	vec3 Row[3], Pdum3;

	// Now get scale and shear.
	for (length_t i = 0; i < 3; ++i)
		for (length_t j = 0; j < 3; ++j)
			Row[i][j] = LocalMatrix[i][j];

	// Compute X scale factor and normalize first row.
	/*scale.x = length(Row[0]);
	Row[0] = detail::scale(Row[0], static_cast<float>(1));
	scale.y = length(Row[1]);
	Row[1] = detail::scale(Row[1], static_cast<float>(1));
	scale.z = length(Row[2]);
	Row[2] = detail::scale(Row[2], static_cast<float>(1));*/
	scale.x = length(Row[0]);
	v3Scale(Row[0], static_cast<float>(1));
	scale.y = length(Row[1]);
	v3Scale(Row[1], static_cast<float>(1));
	scale.z = length(Row[2]);
	v3Scale(Row[2], static_cast<float>(1));

	// At this point, the matrix (in rows[]) is orthonormal.
	// Check for a coordinate system flip.  If the determinant
	// is -1, then negate the matrix and the scaling factors.
#if 0
	Pdum3 = cross(Row[1], Row[2]); // v3Cross(row[1], row[2], Pdum3);
	if (dot(Row[0], Pdum3) < 0)
	{
		for (length_t i = 0; i < 3; i++)
		{
			scale[i] *= static_cast<float>(-1);
			Row[i] *= static_cast<float>(-1);
		}
	}
#endif

	/*rotation.y = asin(-Row[0][2]);
	if (cos(rotation.y) != 0) {
		rotation.x = atan2(Row[1][2], Row[2][2]);
		rotation.z = atan2(Row[0][1], Row[0][0]);
	}
	else {
		rotation.x = atan2(-Row[2][0], Row[1][1]);
		rotation.z = 0;
	}*/


	float s, t, x, y, z, w;

	t = Row[0][0] + Row[1][1] + Row[2][2] + 1.0f;

	if (t > 1e-4)
	{
		s = 0.5f / glm::sqrt(t);
		w = 0.25f / s;
		x = (Row[2][1] - Row[1][2]) * s;
		y = (Row[0][2] - Row[2][0]) * s;
		z = (Row[1][0] - Row[0][1]) * s;
	}
	else if (Row[0][0] > Row[1][1] && Row[0][0] > Row[2][2])
	{
		s = glm::sqrt(1.0f + Row[0][0] - Row[1][1] - Row[2][2]) * 2.0f; // S=4*qx 
		x = 0.25f * s;
		y = (Row[0][1] + Row[1][0]) / s;
		z = (Row[0][2] + Row[2][0]) / s;
		w = (Row[2][1] - Row[1][2]) / s;
	}
	else if (Row[1][1] > Row[2][2])
	{
		s = glm::sqrt(1.0f + Row[1][1] - Row[0][0] - Row[2][2]) * 2.0f; // S=4*qy
		x = (Row[0][1] + Row[1][0]) / s;
		y = 0.25f * s;
		z = (Row[1][2] + Row[2][1]) / s;
		w = (Row[0][2] - Row[2][0]) / s;
	}
	else
	{
		s = glm::sqrt(1.0f + Row[2][2] - Row[0][0] - Row[1][1]) * 2.0f; // S=4*qz
		x = (Row[0][2] + Row[2][0]) / s;
		y = (Row[1][2] + Row[2][1]) / s;
		z = 0.25f * s;
		w = (Row[1][0] - Row[0][1]) / s;
	}

	rotation.x = x;
	rotation.y = y;
	rotation.z = z;
	rotation.w = w;

	return true;
}

void Transform::v3Scale(glm::vec3& v, float desiredLength)
{
	float len = glm::length(v);
	if (len != 0)
	{
		float l = desiredLength / len;
		v[0] *= l;
		v[1] *= l;
		v[2] *= l;
	}
}