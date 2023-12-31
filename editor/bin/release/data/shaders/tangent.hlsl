#vertex
// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 fragPos : FRAG_POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

// constant buffers
cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
{
    matrix projection;
    matrix view;
    matrix viewProjection;
    float3 cameraPos;
}

cbuffer VS_CONSTANT_BUFFER2 : register(b3)
{
    matrix model;
}

// vertex shader 
VS_OUTPUT VSMain(VS_INPUT input)
{
    float4x4 modelViewProj = mul(projection, mul(view, model));
    VS_OUTPUT output;
    output.position = mul(modelViewProj, float4(input.position, 1.0f));

    float4 fp = mul(model, float4(input.position, 1.0f));
    output.fragPos = float3(fp.x, fp.y, fp.z);
    output.normal = normalize(input.normal);
    output.texCoord = input.texCoord;

    return output;
}

#fragment
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float3 fragPos : FRAG_POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    // derivations of the fragment position
    float3 pos_dx = ddx(input.fragPos);
    float3 pos_dy = ddy(input.fragPos);
    // derivations of the texture coordinate
    float2 texC_dx = ddx(input.texCoord);
    float2 texC_dy = ddy(input.texCoord);
    // tangent vector and binormal vector
    float3 tangent = texC_dy.y * pos_dx - texC_dx.y * pos_dy;
    tangent = tangent - input.normal * dot(tangent, input.normal);
    
    return float4(normalize(tangent), 1.0f);
}