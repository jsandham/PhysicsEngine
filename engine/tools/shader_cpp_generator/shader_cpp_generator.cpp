#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <unordered_map>
#include <filesystem>

void write_header(const std::string& name, std::ofstream& out)
{
    out << "//***************************************\n";
    out << "// THIS IS A GENERATED FILE. DO NOT EDIT.\n";
    out << "//***************************************\n";
    out << "#include <string>\n";
    out << "#include \"glsl_shaders.h\"\n";
    out << ("using namespace " + name + ";\n");
}

void write_scope_start(std::ofstream &out)
{
    out << "{\n";
}

void write_scope_end(std::ofstream &out)
{
    out << "}\n";
}

void write_function_declaration(const std::string& functionNamespace, const std::string& functionName, std::ofstream& out)
{
    std::string declaration = "std::string " + functionNamespace + "::" + functionName + "()\n";
    out << declaration;
}

void write_function_body(const std::string& shaderFilepath, std::ofstream& out)
{
    std::ifstream in;
    in.open(shaderFilepath, std::ifstream::in);

    if (!in.is_open())
    {
        return;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();

    out << "return ";

    std::string line;
    while (std::getline(buffer, line))
    {
        out << ("\"" + line + "\\n\"\n");
    }
    out << ";\n";

    //out << buffer.str();
}

void generate_shader_cpp_file()
{
    std::ofstream out;
    out.open("../../src/graphics/GLSL/glsl_shaders.cpp");

    if (!out.is_open())
    {
        return;
    }

    write_header("PhysicsEngine", out);
  
    std::cout << "Generating shader cpp file..." << std::endl;
    
    std::cout << "Reading geometry_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/geometry_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading geometry_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGeometryFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/geometry_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading ssao_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/ssao_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading ssao_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSSAOFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/ssao_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_map_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_map_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthMapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_map_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading shadow_depth_cubemap_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading shadow_depth_cubemap_g.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getShadowDepthCubemapGeometryShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/shadow_depth_cubemap_g.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading color_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/color_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading color_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getColorFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/color_f.glsl", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/screen_quad_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading screen_quad_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getScreenQuadFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/screen_quad_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading sprite_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/sprite_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading sprite_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getSpriteFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/sprite_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading gbuffer_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gbuffer_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading gbuffer_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGBufferFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gbuffer_f.glsl", out);
    write_scope_end(out);
   
    std::cout << "Reading normal_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/normal_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading normal_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getNormalFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/normal_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading position_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/position_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading position_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getPositionFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/position_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading linear_depth_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/linear_depth_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading linear_depth_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLinearDepthFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/linear_depth_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading line_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/line_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading line_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getLineFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/line_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading gizmo_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gizmo_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading gizmo_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGizmoFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/gizmo_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading grid_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/grid_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading grid_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getGridFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/grid_f.glsl", out);
    write_scope_end(out);
    
    std::cout << "Reading standard_v.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardVertexShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/standard_v.glsl", out);
    write_scope_end(out);

    std::cout << "Reading standard_f.glsl" << std::endl;
    write_function_declaration("PhysicsEngine", "getStandardFragmentShader", out);
    write_scope_start(out);
    write_function_body("../../src/graphics/glsl/standard_f.glsl", out);
    write_scope_end(out);

    out.close();
}





enum class TokenType
{
    Add,
    Subtract,
    Multiply,
    Divide,
    Equals,
    OpenBracket,
    ClosedBracket,
    OpenBrace,
    ClosedBrace,
    OpenParen,
    ClosedParen,
    SemiColon,
    Colon,
    String,
    PreprocessorIdentifier,
    Identifier,
    Numeric,
    EndOfStream,
    Unknown
};

enum class PreprocessorTokenType
{
    If,
    Elif,
    Else,
    EndIf,
    Version,
    Include,
    Unknown
};

struct Token
{
    TokenType type;
    size_t length;
    char *text;
};

struct PreprocessorToken
{
    PreprocessorTokenType type;
    bool handled;
};

struct Tokenizer
{
    char *at;
};

std::string tokenTypeToString(TokenType type)
{
    switch (type)
    {
    case TokenType::Add: {
        return "Add";
    }
    case TokenType::Subtract: {
        return "Subtract";
    }
    case TokenType::Multiply: {
        return "Multiply";
    }
    case TokenType::Divide: {
        return "Divide";
    }
    case TokenType::Equals: {
        return "Equals";
    }
    case TokenType::OpenBracket: {
        return "OpenBracket";
    }
    case TokenType::ClosedBracket: {
        return "ClosedBracket";
    }
    case TokenType::OpenBrace: {
        return "OpenBrace";
    }
    case TokenType::ClosedBrace: {
        return "ClosedBrace";
    }
    case TokenType::OpenParen: {
        return "OpenParen";
    }
    case TokenType::ClosedParen: {
        return "CloasedParen";
    }
    case TokenType::SemiColon: {
        return "SemiColon";
    }
    case TokenType::Colon: {
        return "Colon";
    }
    case TokenType::String: {
        return "String";
    }
    case TokenType::PreprocessorIdentifier: {
        return "PreprocessorIdentifier";
    }
    case TokenType::Identifier: {
        return "Identifier";
    }
    case TokenType::Numeric: {
        return "Numeric";
    }
    case TokenType::EndOfStream: {
        return "EndOfStream";
    }
    default: {
        return "Unknown";
    }
    }
}

bool isWhiteSpace(char c)
{
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r')
    {
        return true;
    }

    return false;
}

bool isEndOfLine(char c)
{
    if (c == '\n' || c == '\r')
    {
        return true;
    }

    return false;
}

void eatAllWhiteSpace(Tokenizer* tokenizer)
{
    while (true)
    {
        if (isWhiteSpace(tokenizer->at[0]))
        {
            tokenizer->at++;
        }
        else if (tokenizer->at[0] == '/' && tokenizer->at[1] == '/')
        {
            tokenizer->at += 2;
            while (tokenizer->at[0] != '\0' && !isEndOfLine(tokenizer->at[0]))
            {
                tokenizer->at++;
            }

            tokenizer->at++;
        }
        else if (tokenizer->at[0] == '/' && tokenizer->at[1] == '*')
        {
            tokenizer->at += 2;
            while (tokenizer->at[0] != '\0' && tokenizer->at[0] != '*' && tokenizer->at[1] != '/')
            {
                tokenizer->at++;
            }
            if (tokenizer->at[0] == '*')
            {
                tokenizer->at += 2;
            }

            tokenizer->at++;
        }
        else
        {
            break;
        }
    }
}

bool isAlphebetical(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool isNumeric(char c)
{
    return (c >= '0' && c <= '9');
}

bool isAlphaNumeric(char c)
{
    return isAlphebetical(c) || isNumeric(c);
}

bool isUnderscore(char c)
{
    return (c == '_');
}

Token getToken(Tokenizer* tokenizer)
{
    eatAllWhiteSpace(tokenizer);

    Token token;
    token.length = 1;
    token.text = tokenizer->at;

    switch (*tokenizer->at)
    {
    case '\0': {
        token.type = TokenType::EndOfStream;
        tokenizer->at++;
        break;
    }
    case '+': {
        token.type = TokenType::Add;
        tokenizer->at++;
        break;
    }
    case '-': {
        token.type = TokenType::Subtract;
        tokenizer->at++;
        break;
    }
    case '*': {
        token.type = TokenType::Multiply;
        tokenizer->at++;
        break;
    }
    case '/': {
        token.type = TokenType::Divide;
        tokenizer->at++;
        break;
    }
    case '=': {
        token.type = TokenType::Equals;
        tokenizer->at++;
        break;
    }
    case '[': {
        token.type = TokenType::OpenBracket;
        tokenizer->at++;
        break;
    }
    case ']': {
        token.type = TokenType::ClosedBracket;
        tokenizer->at++;
        break;
    }
    case '{': {
        token.type = TokenType::OpenBrace;
        tokenizer->at++;
        break;
    }
    case '}': {
        token.type = TokenType::ClosedBrace;
        tokenizer->at++;
        break;
    }
    case '(': {
        token.type = TokenType::OpenParen;
        tokenizer->at++;
        break;
    }
    case ')': {
        token.type = TokenType::ClosedParen;
        tokenizer->at++;
        break;
    }
    case ';': {
        token.type = TokenType::SemiColon;
        tokenizer->at++;
        break;
    }
    case ':': {
        token.type = TokenType::Colon;
        tokenizer->at++;
        break;
    }
    case '"': {
        tokenizer->at++; // skip starting "
        token.type = TokenType::String;
        token.text = tokenizer->at;
        while (tokenizer->at[0] != '\0' && tokenizer->at[0] != '"')
        {
            if (tokenizer->at[0] == '\\' && tokenizer->at[1] != '\0')
            {
                tokenizer->at++;
            }
            tokenizer->at++;
            token.length++;
        }
        if (tokenizer->at[0] == '"')
        {
            tokenizer->at++; // skip ending "
        }
        break;
    }
    case '#': {
        token.type = TokenType::PreprocessorIdentifier;
        token.text = tokenizer->at;
        token.length = 1;

        tokenizer->at++;
        while (isAlphebetical(tokenizer->at[0]))
        {
            token.length++;
            tokenizer->at++;
        }
        break;
    }
    default: {
        if (isAlphebetical(tokenizer->at[0]) || isUnderscore(tokenizer->at[0]))
        {
            token.type = TokenType::Identifier;
            token.text = tokenizer->at;
            token.length = 0;
            while (isAlphaNumeric(tokenizer->at[0]) || isUnderscore(tokenizer->at[0]))
            {
                token.length++;
                tokenizer->at++;
            }
            break;
        }
        else if (isNumeric(tokenizer->at[0]))
        {
            token.type = TokenType::Numeric;
            token.text = tokenizer->at;
            token.length = 0;
            while (isNumeric(tokenizer->at[0]))
            {
                token.length++;
                tokenizer->at++;
            }
            break;
        }
        else
        {
            token.type = TokenType::Unknown;
            tokenizer->at++;
            break;
        }
    }
    }

    return token;
}

char* readEntireFileIntoBuffer(const std::string& filename, size_t& bufferLength)
{
    std::ifstream in;
    in.open(filename, std::ifstream::in);

    if (!in.is_open())
    {
        return nullptr;
    }

    std::stringstream ss;
    ss << in.rdbuf();
    in.close();

    std::string str = ss.str();
    int length = str.length();

    char *buffer = (char *)malloc((length + 1) * sizeof(char));
    memcpy(buffer, str.c_str(), length * sizeof(char));

    buffer[length] = '\0';

    bufferLength = length + 1;

    //std::cout << "file contents" << std::endl;
    //std::cout << str << std::endl;

    return buffer;
}

std::vector<Token> performLexographicalAnalysis(char *buffer)
{
    Tokenizer tokenizer;
    tokenizer.at = buffer;

    std::vector<Token> tokens;

    bool done = false;
    while (!done)
    {
        Token token = getToken(&tokenizer);
        tokens.push_back(token);

        switch (token.type)
        {
        case TokenType::EndOfStream:
            done = true;
            break;
        //case TokenType::Unknown:
        //    std::cout << "unknown token" << std::endl;
        //    break;
        default:
            //std::string textstr = std::string(token.text, token.text + token.length);
            //std::cout << "Type: " << tokenTypeToString(token.type) << " length: " << token.length << " text: " << textstr << std::endl;
            break;
        }
    }

    return tokens;
}

struct File
{
    std::string path;
    char *buffer;
    size_t bufferLength;
    std::vector<Token> tokens;
};

bool compare(char* str1, char* str2, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        if (str1[i] != str2[i])
        {
            return false;
        }
    }
    return true;
}

PreprocessorTokenType getPreprocessorTokenType(Token token)
{
    if (token.length == 3)
    {
        if (compare(token.text, "#if", token.length))
        {
            return PreprocessorTokenType::If;
        }
    }
    else if (token.length == 5)
    {
        if (compare(token.text, "#elif", token.length))
        {
            return PreprocessorTokenType::Elif;
        }
        else if (compare(token.text, "#else", token.length))
        {
            return PreprocessorTokenType::Else;
        }
    }
    else if (token.length == 6)
    {
        if (compare(token.text, "#endif", token.length))
        {
            return PreprocessorTokenType::EndIf;
        }
    }
    else if (token.length == 8)
    {
        if (compare(token.text, "#include", token.length))
        {
            return PreprocessorTokenType::Include;    
        }
        else if (compare(token.text, "#version", token.length))
        {
            return PreprocessorTokenType::Version;
        }
    }

    return PreprocessorTokenType::Unknown;
}

bool fileContainsInclude(const File& file)
{
    for (size_t i = 0; i < file.tokens.size(); i++)
    {
        if (file.tokens[i].type == TokenType::PreprocessorIdentifier)
        {
            if (getPreprocessorTokenType(file.tokens[i]) == PreprocessorTokenType::Include)
            {
                std::cout << "pre-processor identifier: " << tokenTypeToString(file.tokens[i].type) << " "
                          << std::string(file.tokens[i].text, file.tokens[i].text + file.tokens[i].length)
                          << " file: " << file.path << std::endl;
                return true;
            }
        }
    }

    return false;
}

void preprocess(const std::string& directoryPath)
{
    if (!std::filesystem::exists(directoryPath))
    {
        return;
    }

    std::vector<File> files;
    std::unordered_map<std::string, File*> filePathToFileMap;

    std::error_code error_code;
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(directoryPath, error_code))
    {
        if (std::filesystem::is_regular_file(entry, error_code) && entry.path().extension() == ".glsl")
        {
            size_t bufferLength;
            char *buffer = readEntireFileIntoBuffer(entry.path().string(), bufferLength);
            if (buffer != nullptr)
            {
                File file;
                file.path = entry.path().string();
                file.buffer = buffer;
                file.bufferLength = bufferLength;

                files.push_back(file);

                std::cout << "File path: " <<  file.path << std::endl;
            }
        }
    }

    for (size_t i = 0; i < files.size(); i++)
    {
        filePathToFileMap[files[i].path] = &files[i];

        files[i].tokens = performLexographicalAnalysis(files[i].buffer);
    }

    std::stack<File*> filesWithIncludes;
    for (size_t i = 0; i < files.size(); i++)
    {
        if (fileContainsInclude(files[i]))
        {
            filesWithIncludes.push(&files[i]);
        }
    }

    while (!filesWithIncludes.empty())
    {
        File *current = filesWithIncludes.top();
        filesWithIncludes.pop();

        std::cout << "current->path: " << current->path << std::endl;

        for (size_t i = 0; i < current->tokens.size(); i++)
        {
            Token *includeToken = &current->tokens[i];
            if (includeToken->type == TokenType::PreprocessorIdentifier)
            {
                if (getPreprocessorTokenType(*includeToken) == PreprocessorTokenType::Include)
                {
                    Token* fileStringToken = &current->tokens[i + 1];
                    if (fileStringToken->type == TokenType::String)
                    {
                        std::string includePath =
                            directoryPath +
                            std::string(fileStringToken->text, fileStringToken->text + fileStringToken->length - 1);

                        std::cout << "AAAAAAAAAAAAAAA " << includePath << std::endl;
                        std::unordered_map<std::string, File *>::iterator it = filePathToFileMap.find(includePath);
                        if (it != filePathToFileMap.end())
                        {
                            File *src = it->second;

                            size_t preIncludeSize = (includeToken->text - current->buffer);
                            size_t includeSize = src->bufferLength - 1; // skip EndOfStream character
                            size_t postIncludeSize = current->buffer + current->bufferLength -
                                                     (fileStringToken->text + fileStringToken->length);
                            size_t newBufferLength = preIncludeSize + includeSize + postIncludeSize;
                            std::cout << "preInclude size: " << preIncludeSize << " include size: " << includeSize
                                      << " post include size: " << postIncludeSize
                                      << " bufferLength: " << current->bufferLength
                                      << " total size: " << newBufferLength
                                      << std::endl;


                            char *newBuffer = (char *)malloc(newBufferLength * sizeof(char));

                            size_t start = 0;
                            memcpy(newBuffer + start, current->buffer, preIncludeSize * sizeof(char));
                            start += preIncludeSize;
                            memcpy(newBuffer + start, src->buffer, includeSize * sizeof(char));
                            start += includeSize;
                            memcpy(newBuffer + start, fileStringToken->text + fileStringToken->length,
                                   postIncludeSize * sizeof(char));
                            start += postIncludeSize;

                            std::cout << "start: " << start << std::endl;

                            std::string test = std::string(newBuffer, newBuffer + start);
                            std::cout << "test: " << test << std::endl;

                            //// skip include, string, and end of stream
                            //std::vector<Token> newTokens(current->tokens.size() + src->tokens.size() - 3);
                            //
                            //size_t index = 0;
                            //size_t base_offset = 0;
                            //for (size_t j = 0; j < i; j++)
                            //{
                            //    size_t offset = base_offset + current->tokens[j].text - current->buffer;

                            //    newTokens[index].type = current->tokens[j].type;
                            //    newTokens[index].length = current->tokens[j].length;
                            //    newTokens[index].text = newBuffer + offset;

                            //    index++;
                            //}

                            //base_offset += current->tokens[i].text - current->buffer;
                            //std::cout << "base_offset: " << base_offset << std::endl;

                            //// skip end of stream token
                            //for (size_t j = 0; j < src->tokens.size() - 1; j++)
                            //{
                            //    size_t offset = base_offset + src->tokens[j].text - src->buffer;

                            //    newTokens[index].type = src->tokens[j].type;
                            //    newTokens[index].length = src->tokens[j].length;
                            //    newTokens[index].text = newBuffer + offset;

                            //    index++;
                            //}

                            //base_offset += src->tokens[src->tokens.size() - 1].text - src->buffer;

                            //std::cout << "base_offset: " << base_offset << std::endl;

                            //for (size_t j = i + 2; j < current->tokens.size(); j++)
                            //{
                            //    size_t offset = base_offset + current->tokens[j].text - current->tokens[i + 2].text;


                            //    newTokens[index].type = current->tokens[j].type;
                            //    newTokens[index].length = current->tokens[j].length;
                            //    newTokens[index].text = newBuffer + offset;

                            //    index++;
                            //}

                            //std::cout << "new tokens size: " << newTokens.size() << " index: " << index << std::endl;







                            free(current->buffer);
                            current->buffer = newBuffer;
                            current->bufferLength = newBufferLength;
                            current->tokens = performLexographicalAnalysis(current->buffer);

                            break;
                        }
                        else
                        {
                            std::cout << "Could not find include path in map" << std::endl;
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < current->tokens.size(); i++)
        {
            std::cout << "current->tokens[i]: " << tokenTypeToString(current->tokens[i].type) << " "
                      << std::string(current->tokens[i].text, current->tokens[i].text + current->tokens[i].length)
                      << std::endl;
        }

        if (fileContainsInclude(*current))
        {
            filesWithIncludes.push(current);
        }
    }

    for (size_t i = 0; i < files.size(); i++)
    {
        free(files[i].buffer);
    }
}




int main()
{
    preprocess("../../src/graphics/glsl/");

    //preprocessor("../../src/graphics/glsl/grid_v.glsl");

    generate_shader_cpp_file();
	return 0;
}