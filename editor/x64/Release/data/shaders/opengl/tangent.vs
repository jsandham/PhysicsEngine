layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    vec3 cameraPos;
}Camera;
uniform mat4 model;
in vec3 position;
in vec3 normal;
in vec2 texCoord;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = normalize(normal);
    TexCoord = texCoord;
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
}