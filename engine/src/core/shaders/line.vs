STRINGIFY(
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 mvp;
out vec4 Color;
void main()
{
   Color = color;
	gl_Position = mvp * vec4(position, 1.0);
}
)