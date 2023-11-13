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
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
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
"vector aVec;\n"
"\n"
"cbuffer Test1 \n"
"{\n"
"    matrix wvp;\n"
"    //vector cameraPos;\n"
"    float alpha;\n"
"};\n"
"\n"
"cbuffer Test2\n"
"{\n"
"    matrix view;\n"
"    // vector cameraPos;\n"
"    // float alpha;\n"
"};\n"
"\n"
"cbuffer Test3\n"
"{\n"
"    matrix model;\n"
"    // vector cameraPos;\n"
"    // float alpha;\n"
"};\n"
"\n"
"// vertex shader \n"
"VS_OUTPUT VSMain(VS_INPUT input)\n"
"{\n"
"    VS_OUTPUT output;\n"
"    output.position = mul(worldViewProjection, float4(input.position, 1.0));\n"
"    output.normal = input.normal;\n"
"\n"
"    float4 temp1 = mul(wvp, float4(input.position, 1.0));\n"
"    float4 temp2 = mul(view, float4(input.position, 1.0));\n"
"    float4 temp3 = mul(model, float4(input.position, 1.0));\n"
"\n"
"    output.position.x = alpha * temp1.x;\n"
"    output.position.y = alpha * temp2.y;\n"
"    output.position.z = alpha * temp3.z;\n"
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
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getGridVertexShader()
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
"    float3 normal : NORMAL;\n"
"};\n"
"\n"
"// pixel shader\n"
"float4 PSMain(PS_INPUT input) : SV_TARGET\n"
"{\n"
"    return float4(input.normal, 1.0);\n"
"}\n";
}
std::string hlsl::getLineVertexShader()
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
std::string hlsl::getNormalFragmentShader()
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
std::string hlsl::getScreenQuadFragmentShader()
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
std::string hlsl::getScreenQuadVertexShader()
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
