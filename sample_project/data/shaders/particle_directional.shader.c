VERTEX:

#version 330 core

in vec3 position;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float pointRadius;
uniform float pointScale; 

void main()
{
    vec3 pos = vec3(view * model * vec4(position.x, position.y, position.z, 1.0f));

    gl_PointSize = 2.0f;//pointRadius * (pointScale / length(pos));

	gl_Position = projection * view * model * vec4(position.x, position.y, position.z, 1.0f);
}


// #version 330 core

// layout (std140) uniform CameraBlock
// {
// 	mat4 projection;
// 	mat4 view;
// 	vec3 cameraPos;
// };

// layout (std140) uniform LightBlock
// {
// 	mat4 lightProjection[5];
// 	mat4 lightView[5];
// };

// in vec3 position;
// in vec2 atexture;

// out vec2 TexCoord;

// uniform mat4 model;
// uniform float pointRadius;
// uniform float pointScale; 

// void main()
// {
//     vec3 pos = vec3(view * model * vec4(position.x, position.y, position.z, 1.0f));

//     gl_PointSize = pointRadius * (pointScale / length(pos));

// 	gl_Position = projection * view * model * vec4(position.x, position.y, position.z, 1.0f);

//     TexCoord = atexture;
// }


FRAGMENT:

#version 330 core

out vec4 FragColor;

void main(void) {
    FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}


// #version 330 core

// struct Material
// {
//     float shininess;
//     vec3 ambient;
//     vec3 diffuse;
//     vec3 specular;
// };

// struct DirLight
// {
//     vec3 direction;

//     vec3 ambient;
//     vec3 diffuse;
//     vec3 specular;
// };

// in vec2 TexCoord;

// out vec4 FragColor;

// uniform Material material;
// uniform DirLight dirLight;
// uniform sampler2D texture0;

// vec3 CalcDirLight(Material material, DirLight light, vec3 normal);

// void main(void) {
//     vec3 Normal;

//     Normal.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
//     float mag = dot(Normal.xy, Normal.xy);

//     if (mag > 1.0f) discard;

//     Normal.z = sqrt(1.0-mag);

// 	vec3 finalLight = CalcDirLight(material, dirLight, Normal);

//     FragColor = texture(texture0, TexCoord.xy) * vec4(finalLight, 1.0f);
// }


// vec3 CalcDirLight(Material material, DirLight light, vec3 normal)
// {
// 	vec3 lightDir = normalize(-light.direction);

//     float diffuseStrength = max(dot(normal, lightDir), 0.0);

//     vec3 diffuse = light.diffuse * material.diffuse * diffuseStrength;

//     return diffuse;
// }