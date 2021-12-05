#version 430 core
in vec3 FragPos;
out vec4 FragColor;
void main()
{
  FragColor = vec4(FragPos, 1);
}