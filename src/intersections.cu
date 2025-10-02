#include "intersections.h"
#include <glm/gtx/intersect.hpp>
#include "bvh.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    //if (!outside)
    //{
    //    normal = -normal;
    //}

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest
(Geom mesh,
    Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float tmax = 1e38f;
    float tmin = tmax;
    glm::vec3 baryNorm;
    glm::vec2 new_uv;

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    for (int i = mesh.meshStartIdx; i < mesh.meshEndIdx + 1; ++i) {
        Triangle curr_tri = triangles[i];
        glm::vec3 baryPos;
        bool intersected = glm::intersectRayTriangle(rt.origin, rt.direction,
            curr_tri.vertices[0], curr_tri.vertices[1], curr_tri.vertices[2], baryPos);

        if (intersected) {
            float t = baryPos.z;
            if (t > 0 && t < tmin) {
                // get closest intersection
                tmin = t;
                glm::vec3 baryNorm_val = ((1 - baryPos.x - baryPos.y) * curr_tri.normals[0])
                    + (baryPos.x * curr_tri.normals[1]) + (baryPos.y * curr_tri.normals[2]);
                baryNorm = glm::normalize(baryNorm_val);

                //textures
                if (mesh.hasTexture) {
                    new_uv = (1.0f - baryPos.x - baryPos.y) * curr_tri.uvs[0]
                        + baryPos.x * curr_tri.uvs[1] + baryPos.y * curr_tri.uvs[2];
                }
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(baryNorm, 0.0f)));
        uv = new_uv;
        return glm::length(r.origin - intersectionPoint);
    }

    outside = false;
    return -1;
}

__host__ __device__ float BVHIntersectionTest
(Geom mesh,
    Triangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside,
    BVHNode* bvhNodes)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    float tmax = 1e38f;
    float tmin = tmax;
    glm::vec3 baryNorm;

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    int BVHnode_stack[64];
    int stack_ptr = 0;
    BVHnode_stack[stack_ptr++] = mesh.bvhRootIdx;

    while (stack_ptr > 0) {
        BVHNode curr_node = bvhNodes[BVHnode_stack[--stack_ptr]];
        float intersect_box = IntersectAABB(rt, curr_node.aabbMin, curr_node.aabbMax);
        if (intersect_box < 0) {
            continue;
        }

        if (curr_node.triangleCount > 0) {
            for (int i = curr_node.firstTriangleIdx; i < curr_node.firstTriangleIdx + curr_node.triangleCount; ++i) {
                Triangle curr_tri = triangles[i];
                glm::vec3 baryPos;
                bool intersected = glm::intersectRayTriangle(rt.origin, rt.direction,
                    curr_tri.vertices[0], curr_tri.vertices[1], curr_tri.vertices[2], baryPos);
                if (intersected) {
                    float t = baryPos.z;
                    if (t > 0 && t < tmin) {
                        // get closest intersection
                        tmin = t;
                        glm::vec3 baryNorm_val = ((1 - baryPos.x - baryPos.y) * curr_tri.normals[0])
                            + (baryPos.x * curr_tri.normals[1]) + (baryPos.y * curr_tri.normals[2]);
                        baryNorm = glm::normalize(baryNorm_val);
                    }
                }
            }
        }
        else {
            BVHnode_stack[stack_ptr++] = curr_node.leftChild;
            BVHnode_stack[stack_ptr++] = curr_node.rightChild;
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(baryNorm, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    outside = false;
    return -1;
}

__host__ __device__ float IntersectAABB(const Ray ray, const glm::vec3 bmin, const glm::vec3 bmax)
{
    // slab method
    glm::vec3 ro = ray.origin;
    glm::vec3 rd = ray.direction;
    float tmin = -1e38f;
    float tmax = 1e38f;

    for (int i = 0; i < 3; i++) {
        float inverse_dir = 1.0f / rd[i];
        float t0 = (bmin[i] - ro[i]) * inverse_dir;
        float t1 = (bmax[i] - ro[i]) * inverse_dir;
        if (inverse_dir < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        tmin = glm::max(tmin, t0);
        tmax = glm::min(tmax, t1);
        if (tmax < tmin) {
            return -1;
        }
    }
    if (tmax < 0.0f) return -1.0f;
    return tmin;
    //return fmaxf(tmax, fabsf(tmin));

}