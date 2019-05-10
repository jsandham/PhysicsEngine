// #include "../../include/graphics.VertexBuffer.h"

// using namespace PhysicsEngine;

// VertexBuffer::VertexBuffer()
// {
//     glGenVertexArrays(1, &m_vao);
//     glGenBuffers(1, &m_vbo);
//     glGenBuffers(1, &m_ibo);
//     glBindVertexArray(m_vao);
//     glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
//     vertex_type::EnableVertexAttribArray(); 
//     glBindBuffer(GL_ARRAY_BUFFER, 0);
//     glBindVertexArray(0);
// }

// VertexBuffer::~VertexBuffer()
// {
//     glDeleteVertexArrays(1, &m_vao);
//     glDeleteBuffers(1, &m_vbo);
//     glDeleteBuffers(1, &m_ibo);
// }