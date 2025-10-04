#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int numTriangles;
	int meshStartIdx;
    int meshEndIdx;
	int bvhRootIdx;
    int numNodes;
	int textureIndex = -1;
	bool hasTexture;
	int normalIndex = -1;
    int hasNormal;
};

struct Triangle
{
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
    int materialid;
    glm::vec3 centroid;
	glm::vec2 uvs[3];
	glm::vec3 tangent;
	glm::vec3 bitangent;
};

struct Texture
{
    int width;
    int height;
    glm::vec3 color;
	unsigned char* data;
    int channels;
};

struct BVHNode {
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    int leftChild;
    int rightChild;
    int firstTriangleIdx;
    int triangleCount;
    bool isLeaf() { return triangleCount > 0; }
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    bool isProcedural = false;
    glm::vec3 checker_color1;
	glm::vec3 checker_color2;
    float checker_scale;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int textureId = -1;
  glm::vec2 uv;
  int normalId = -1;
  glm::vec3 tangent;
  glm::vec3 bitangent;
};
