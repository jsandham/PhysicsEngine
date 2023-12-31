//***************************************
// THIS IS A GENERATED FILE. DO NOT EDIT.
//***************************************
#include <string>
#include "hlsl_shaders.h"
using namespace hlsl;
std::string hlsl::getColorFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"};\n"
"\n"
"struct Material\n"
"{\n"
"    uint4 color;\n"
"};\n"
"\n"
"cbuffer VS_CONSTANT_BUFFER : register(b3)\n"
"{\n"
"    Material color;\n"
"}\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(color.color.r / 255.0f, color.color.g / 255.0f, color.color.b / 255.0f, color.color.a / 255.0f);\n"
"}\n";
}
std::string hlsl::getColorInstancedFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getColorInstancedVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getColorVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer CAMERA_CONSTANT_BUFFER : register(b0)\n"
"{\n"
"    matrix projection;\n"
"    matrix view;\n"
"    matrix viewProjection;\n"
"    float3 cameraPos;\n"
"}\n"
"\n"
"cbuffer VS_CONSTANT_BUFFER : register(b3)\n"
"{\n"
"    matrix model;\n"
"}\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    float4x4 modelViewProj = mul(projection, mul(view, model));\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(modelViewProj, float4(input.position, 1.0f));\n"
"\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getGBufferFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getGBufferVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getGeometryFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getGeometryVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getGizmoFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getGizmoInstancedFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getGizmoInstancedVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getGizmoVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getGridFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float4 color : COLOR;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return input.color;\n"
"}\n";
}
std::string hlsl::getGridVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float4 color : COLOR;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer VS_CONSTANT_BUFFER : register(b3)\n"
"{\n"
"    matrix mvp;\n"
"    float4 color;\n"
"}\n"
"\n"
"// vertex shader\n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(mvp, float4(input.position, 1.0f));\n"
"    output.color = color;\n"
"\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getLinearDepthFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getLinearDepthInstancedFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getLinearDepthInstancedVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getLinearDepthVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getLineFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float4 color : COLOR;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return input.color;\n"
"}\n";
}
std::string hlsl::getLineVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float4 color : COLOR;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float4 color : COLOR;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer VS_CONSTANT_BUFFER : register(b3)\n"
"{\n"
"    matrix mvp;\n"
"}\n"
"\n"
"// vertex shader\n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(mvp, float4(input.position, 1.0f));\n"
"    output.color = input.color;\n"
"    \n"
"    return output;\n"
"}\n";
}
std::string hlsl::getNormalFragmentShader()
{
return "// INPUT structures\n"
"struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getNormalInstancedFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getNormalInstancedVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getNormalVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer CAMERA_CONSTANT_BUFFER : register(b0)\n"
"{\n"
"    matrix projection;\n"
"    matrix view;\n"
"    matrix viewProjection;\n"
"    float3 cameraPos;\n"
"}\n"
"\n"
"cbuffer VS_CONSTANT_BUFFER2 : register(b3)\n"
"{\n"
"    matrix model;\n"
"}\n"
"\n"
"// vertex shader\n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    float4x4 modelViewProj = mul(projection, mul(view, model));\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(modelViewProj, float4(input.position, 1.0f));\n"
"    output.normal = input.normal;\n"
"\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getOcclusionMapFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(1.0, 0.0, 0.0, 1.0);\n"
"}\n";
}
std::string hlsl::getOcclusionMapVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getPositionFragmentShader()
{
return "// INPUT structures\n"
"struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 fragPos : FRAG_POSITION;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.fragPos, 1.0f);\n"
"}\n";
}
std::string hlsl::getPositionInstancedFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getPositionInstancedVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getPositionVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 fragPos : FRAG_POSITION;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer CAMERA_CONSTANT_BUFFER : register(b0)\n"
"{\n"
"    matrix projection;\n"
"    matrix view;\n"
"    matrix viewProjection;\n"
"    float3 cameraPos;\n"
"}\n"
"\n"
"cbuffer VS_CONSTANT_BUFFER2 : register(b3)\n"
"{\n"
"    matrix model;\n"
"}\n"
"\n"
"// vertex shader\n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    float4 worldPos = mul(model, float4(input.position, 1.0f));\n"
"    float4x4 modelViewProj = mul(projection, mul(view, model));\n"
"    \n"
"    VS_OUTPUT output;\n"
"    output.position = mul(modelViewProj, float4(input.position, 1.0f));\n"
"    output.fragPos = worldPos.xyz;\n"
"\n"
"    return output;\n"
"}\n"
"\n";
}
std::string hlsl::getScreenQuadFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float2 texCoord : TEXCOORD;\n"
"};\n"
"\n"
"Texture2D screenTexture;\n"
"SamplerState sampleState\n"
"{\n"
"    Filter = MIN_MAG_MIP_LINEAR;\n"
"    AddressU = Wrap;\n"
"    AddressV = Wrap;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    float3 col = screenTexture.Sample(sampleState, input.texCoord).rgb;\n"
"    return float4(col, 1.0f);\n"
"}\n";
}
std::string hlsl::getScreenQuadVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float2 position : POSITION;\n"
"    float2 texCoord : TEXCOORD;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float2 texCoord : TEXCOORD;\n"
"};\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = float4(input.position.x, input.position.y, 0.0f, 1.0f);\n"
"    output.texCoord = input.texCoord;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getShadowDepthCubemapFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getShadowDepthCubemapGeometryShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getShadowDepthCubemapVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getShadowDepthMapFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getShadowDepthMapVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getSpriteFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getSpriteVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getSSAOFragmentShader()
{
return "struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getSSAOVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// uniforms : external parameters\n"
"matrix worldViewProjection;\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"    return output;\n"
"}\n";
}
std::string hlsl::getStandardFragmentShader()
{
return "// INPUT structures\n"
"struct PS_INPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"    float3 fragPos : FRAGPOSITION;\n"
"    float3 cameraPos : CAMERAPOSITION;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer PS_CONSTANT_BUFFER : register(b4)\n"
"{\n"
"    float3 lightDirection;\n"
"    float3 color;\n"
"}\n"
"\n"
"float3 CalcDirLight(float3 normal2, float3 viewDir)\n"
"{\n"
"    float3 norm = normalize(normal2);\n"
"    float3 lightDir = normalize(lightDirection);\n"
"    float3 reflectDir = reflect(-lightDir, norm);\n"
"    float diffuseStrength = max(dot(norm, lightDir), 0.0);\n"
"    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 1.0f);\n"
"    float3 ambient = float3(0.7, 0.7, 0.7);\n"
"    float3 diffuse = float3(1.0, 1.0, 1.0) * diffuseStrength;\n"
"    float3 specular = float3(0.7, 0.7, 0.7) * specularStrength;\n"
"\n"
"    return (ambient + diffuse + specular);\n"
"}\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    float3 viewDir = normalize(input.cameraPos - input.fragPos);\n"
"    return float4(CalcDirLight(input.normal, viewDir) * color, 1.0f);\n"
"}\n";
}
std::string hlsl::getStandardVertexShader()
{
return "// INPUT/OUTPUT structures\n"
"struct VS_INPUT\n"
"{\n"
"    float3 position : POSITION;\n"
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"struct VS_OUTPUT\n"
"{\n"
"    float4 position : SV_POSITION;\n"
"    float3 normal : NORMAL;\n"
"    float3 fragPos : FRAGPOSITION;\n"
"    float3 cameraPos : CAMERAPOSITION;\n"
"};\n"
"\n"
"// constant buffers\n"
"cbuffer CAMERA_CONSTANT_BUFFER : register(b0)\n"
"{\n"
"    matrix projection;\n"
"    matrix view;\n"
"    matrix viewProjection;\n"
"    float3 cameraPos;\n"
"}\n"
"\n"
"cbuffer VS_CONSTANT_BUFFER : register(b3)\n"
"{\n"
"    matrix model;\n"
"}\n"
"\n"
"// vertex shader\n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    float4x4 modelViewProj = mul(projection, mul(view, model));\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(modelViewProj, float4(input.position, 1.0f));\n"
"    output.normal = input.normal;\n"
"    output.cameraPos = cameraPos;\n"
"    float4 fp = mul(model, float4(input.position, 1.0f));\n"
"    output.fragPos = float3(fp.x, fp.y, fp.z);\n"
"\n"
"    return output;\n"
"}\n";
}
