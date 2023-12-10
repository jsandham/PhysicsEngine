#vertex
// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 fragPos : FRAGPOSITION;
    float3 cameraPos : CAMERAPOSITION;
};

// uniforms : external parameters
//cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
//{
//    float3 cameraPos;
//    matrix view;
//    matrix projection;
//    matrix viewProjection;
//}

//cbuffer VS_CONSTANT_BUFFER2 : register(b3)
//{
//    matrix model;
//}

cbuffer VS_CONSTANT_BUFFER : register(b3)
{
    matrix projection;
    matrix view;
    matrix model;
    float3 cameraPos;
}

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    float4x4 modelViewProj = mul(projection, mul(view, model));
    VS_OUTPUT output;
    output.position = mul(modelViewProj, float4(input.position, 1.0f));
    output.normal = input.normal;
    output.cameraPos = cameraPos;
    float4 fp = mul(model, float4(input.position, 1.0f));
    output.fragPos = float3(fp.x, fp.y, fp.z);

    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float3 fragPos : FRAGPOSITION;
    float3 cameraPos : CAMERAPOSITION;
};

cbuffer PS_CONSTANT_BUFFER : register(b4)
{
    float3 lightDirection;
    float3 color;
}

float3 CalcDirLight(float3 normal2, float3 viewDir)
{
    float3 norm = normalize(normal2);
    float3 lightDir = normalize(lightDirection);
    float3 reflectDir = reflect(-lightDir, norm);
    float diffuseStrength = max(dot(norm, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 1.0f);
    float3 ambient = float3(0.7, 0.7, 0.7);
    float3 diffuse = float3(1.0, 1.0, 1.0) * diffuseStrength;
    float3 specular = float3(0.7, 0.7, 0.7) * specularStrength;

    return (ambient + diffuse + specular);
}

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    float3 viewDir = normalize(input.cameraPos - input.fragPos);
    return float4(CalcDirLight(input.normal, viewDir) * color, 1.0f);
}