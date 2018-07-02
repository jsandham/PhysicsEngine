#version 330 core

#define NUM_OF_CASCADES 5

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
};

layout (std140) uniform ShadowBlock
{
	mat4 lightProjection[NUM_OF_CASCADES];
	mat4 lightView[NUM_OF_CASCADES];
	float cascadeEnds[NUM_OF_CASCADES];
	float farPlane;
};

uniform mat4 model;

in vec3 position;
in vec3 normal;

out float FarPlane;
out float ClipSpaceZ;
out float CascadeEnds[NUM_OF_CASCADES];
out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec4 FragPosLightSpace[NUM_OF_CASCADES];

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * normal);
    
    FarPlane = farPlane;

	ClipSpaceZ = gl_Position.z;

	for(int i = 0; i < NUM_OF_CASCADES; i++){
		FragPosLightSpace[i] = lightProjection[i] * lightView[i] * vec4(FragPos, 1.0f);
		CascadeEnds[i] = cascadeEnds[i];
	}

	CameraPos = cameraPos;
}