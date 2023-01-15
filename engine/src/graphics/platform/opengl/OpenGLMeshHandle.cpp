#include "../../../../include/graphics/platform/opengl/OpenGLMeshHandle.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLMeshHandle::OpenGLMeshHandle()
{
    CHECK_ERROR(glGenVertexArrays(1, &mVao));
    CHECK_ERROR(glBindVertexArray(mVao));

    mVbo[0] = VertexBuffer::create();
    mVbo[1] = VertexBuffer::create();
    mVbo[2] = VertexBuffer::create();
    mVbo[3] = VertexBuffer::create();
    mVbo[4] = VertexBuffer::create();

    // set attribute pointers for vertces
    mVbo[0]->bind();
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));

    // set attribute pointers for normals
    mVbo[1]->bind();
    CHECK_ERROR(glEnableVertexAttribArray(1));
    CHECK_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));

    // set attribute pointers for texcoords
    mVbo[2]->bind();
    CHECK_ERROR(glEnableVertexAttribArray(2));
    CHECK_ERROR(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0));

    // set attribute pointers for matrix (4 times vec4)
    mVbo[3]->bind();
    CHECK_ERROR(glEnableVertexAttribArray(3));
    CHECK_ERROR(glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0));
    CHECK_ERROR(glEnableVertexAttribArray(4));
    CHECK_ERROR(glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4))));
    CHECK_ERROR(glEnableVertexAttribArray(5));
    CHECK_ERROR(glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4))));
    CHECK_ERROR(glEnableVertexAttribArray(6));
    CHECK_ERROR(glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4))));

    CHECK_ERROR(glVertexAttribDivisor(3, 1));
    CHECK_ERROR(glVertexAttribDivisor(4, 1));
    CHECK_ERROR(glVertexAttribDivisor(5, 1));
    CHECK_ERROR(glVertexAttribDivisor(6, 1));

    // instancing colors vbo
    mVbo[4]->bind();
    CHECK_ERROR(glEnableVertexAttribArray(7));
    CHECK_ERROR(glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void *)0));

    CHECK_ERROR(glVertexAttribDivisor(7, 1));

    CHECK_ERROR(glBindVertexArray(0));

























    //glGenVertexArrays(1, vao);
    //glBindVertexArray(*vao);
    //glGenBuffers(1, vbo0);      // vertex vbo
    //glGenBuffers(1, vbo1);      // normals vbo
    //glGenBuffers(1, vbo2);      // texcoords vbo
    //glGenBuffers(1, model_vbo); // instance model vbo
    //glGenBuffers(1, color_vbo); // instance color vbo

    //glBindVertexArray(*vao);
    //glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    //glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    //glEnableVertexAttribArray(0);
    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    //glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    //glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
    //glEnableVertexAttribArray(1);
    //glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    //glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
    //glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_DYNAMIC_DRAW);
    //glEnableVertexAttribArray(2);
    //glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

    //// instancing model matrices vbo
    //glBindBuffer(GL_ARRAY_BUFFER, *model_vbo);
    //glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);
    //// set attribute pointers for matrix (4 times vec4)
    //glEnableVertexAttribArray(3);
    //glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0);
    //glEnableVertexAttribArray(4);
    //glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4)));
    //glEnableVertexAttribArray(5);
    //glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4)));
    //glEnableVertexAttribArray(6);
    //glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4)));

    //glVertexAttribDivisor(3, 1);
    //glVertexAttribDivisor(4, 1);
    //glVertexAttribDivisor(5, 1);
    //glVertexAttribDivisor(6, 1);

    //// instancing colors vbo
    //glBindBuffer(GL_ARRAY_BUFFER, *color_vbo);
    //glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::vec4), NULL, GL_DYNAMIC_DRAW);
    //glEnableVertexAttribArray(7);
    //glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void *)0);

    //glVertexAttribDivisor(7, 1);

    //glBindVertexArray(0);

    //Graphics::checkError(__LINE__, __FILE__);
}

OpenGLMeshHandle::~OpenGLMeshHandle()
{
    delete mVbo[0];
    delete mVbo[1];
    delete mVbo[2];
    delete mVbo[3];
    delete mVbo[4];

    CHECK_ERROR(glDeleteVertexArrays(1, &mVao));
}

void OpenGLMeshHandle::bind()
{
    CHECK_ERROR(glBindVertexArray(mVao));
}

void OpenGLMeshHandle::unbind()
{
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::setData(void *data, size_t offset, size_t size, MeshVBO meshVBO)
{
    VertexBuffer *buffer = mVbo[static_cast<int>(meshVBO)];
    buffer->bind();
    if (buffer->getSize() < (offset + size))
    {
        buffer->resize(size);
    }
    buffer->setData(data, offset, size);
    buffer->unbind();
}

void OpenGLMeshHandle::draw()
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, 0, (mVbo[0]->getSize() / sizeof(float)) / 3));
    CHECK_ERROR(glBindVertexArray(0));
}

VertexBuffer *OpenGLMeshHandle::getVBO(MeshVBO meshVBO)
{
    return mVbo[static_cast<int>(meshVBO)];
}

unsigned int OpenGLMeshHandle::getVAO()
{
    return mVao;
}