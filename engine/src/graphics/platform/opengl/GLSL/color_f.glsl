#version 430 core
struct Material
{
    uvec4 color;
};
uniform Material material;
out vec4 FragColor;
void main()
{
    FragColor = vec4(material.color.r / 255.0f, material.color.g / 255.0f,
                      material.color.b / 255.0f, material.color.a / 255.0f);
}