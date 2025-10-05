#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"
#include "stb_image.h"
#include "texture_indirect_functions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define ANTIALIASING 1
#define SORT_MATERIALS 0
#define STREAM_COMPACTION 1
#define BVH 1
#define DOF 0

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static Triangle* dev_triangles = NULL;
static int* dev_triangles_idx = NULL;
static BVHNode* dev_bvhNodes = NULL;

static Texture* dev_textures = NULL;
static cudaTextureObject_t* dev_texture_objects = NULL;
static std::vector<cudaArray_t> dev_cuda_texture_data;
static std::vector<cudaTextureObject_t> hst_texture_objects;

static Texture* dev_normals = NULL;
static cudaTextureObject_t* dev_normal_objects = NULL;
static std::vector<cudaArray_t> dev_normal_data;
static std::vector<cudaTextureObject_t> hst_normal_objects;

static Texture* dev_env_maps = NULL;
static cudaTextureObject_t* dev_env_objects = NULL;
static std::vector<cudaArray_t> dev_env_data;
static std::vector<cudaTextureObject_t> hst_env_objects;


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles_idx, scene->triangles_idx.size() * sizeof(int));
    cudaMemcpy(dev_triangles_idx, scene->triangles_idx.data(), scene->triangles_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    //textures
    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
	dev_cuda_texture_data.resize(scene->textures.size());
    hst_texture_objects.resize(scene->textures.size());
    for (int i = 0; i < scene->textures.size(); i++) {
		Texture curr_tex = scene->textures[i];
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaMallocArray(&dev_cuda_texture_data[i], &channelDesc, scene->textures[i].width, scene->textures[i].height);
        cudaMemcpyToArray(dev_cuda_texture_data[i], 0, 0, scene->textures[i].data, scene->textures[i].channels * scene->textures[i].width * scene->textures[i].height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_cuda_texture_data[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&hst_texture_objects[i], &resDesc, &texDesc, NULL);
    }
    cudaMalloc(&dev_texture_objects, scene->textures.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_texture_objects, hst_texture_objects.data(), scene->textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    //bump map
    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(Texture));
    cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    dev_normal_data.resize(scene->normals.size());
    hst_normal_objects.resize(scene->normals.size());
    for (int i = 0; i < scene->normals.size(); i++) {
        Texture curr_normal = scene->normals[i];
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaMallocArray(&dev_normal_data[i], &channelDesc, curr_normal.width, curr_normal.height);
        cudaMemcpyToArray(dev_normal_data[i], 0, 0, curr_normal.data, curr_normal.channels * curr_normal.width * curr_normal.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_normal_data[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&hst_normal_objects[i], &resDesc, &texDesc, NULL);
    }
    cudaMalloc(&dev_normal_objects, scene->normals.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_normal_objects, hst_normal_objects.data(), scene->normals.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    //environment map
    cudaMalloc(&dev_env_maps, scene->env_maps.size() * sizeof(Texture));
    cudaMemcpy(dev_env_maps, scene->env_maps.data(), scene->env_maps.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    dev_env_data.resize(scene->env_maps.size());
    hst_env_objects.resize(scene->env_maps.size());

    for (int i = 0; i < scene->env_maps.size(); i++) {
        Texture curr_env = scene->env_maps[i];
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaMallocArray(&dev_env_data[i], &channelDesc, curr_env.width, curr_env.height);
        cudaMemcpyToArray(dev_env_data[i], 0, 0, curr_env.data, curr_env.channels * curr_env.width * curr_env.height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_env_data[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeClamp;   // better for env maps
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        cudaCreateTextureObject(&hst_env_objects[i], &resDesc, &texDesc, NULL);
    }
    cudaMalloc(&dev_env_objects, scene->env_maps.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_env_objects, hst_env_objects.data(), scene->env_maps.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);



    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
	cudaFree(dev_triangles_idx);
    cudaFree(dev_bvhNodes);
	cudaFree(dev_textures);
	cudaFree(dev_env_maps);
	cudaFree(dev_normals);

    for (int i = 0; i < hst_texture_objects.size(); i++) {
		cudaDestroyTextureObject(hst_texture_objects[i]);
        cudaFreeArray(dev_cuda_texture_data[i]);
	}
    cudaFree(dev_texture_objects);

    for (int i = 0; i < hst_normal_objects.size(); i++) {
        cudaDestroyTextureObject(hst_normal_objects[i]);
        cudaFreeArray(dev_normal_data[i]);
    }
    cudaFree(dev_normal_objects);

    for (int i = 0; i < hst_env_objects.size(); i++) {
        cudaDestroyTextureObject(hst_env_objects[i]);
        cudaFreeArray(dev_env_data[i]);
    }
    cudaFree(dev_env_objects);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // TODO: implement antialiasing by jittering the ray
#if ANTIALIASING
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + (u01(rng) - 0.5f) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + (u01(rng) - 0.5f) - (float)cam.resolution.y * 0.5f)
        );  
#endif

#if DOF
		// from PBR Chapter 5.2
        glm::vec3 pLens;
        pLens.x = cam.lensRadius * u01(rng);
		pLens.y = cam.lensRadius * u01(rng);
        pLens.z = 0.0f;
        //float ft = cam.focalDistance / segment.ray.direction.z;
		float ft = cam.focalDistance / glm::dot(segment.ray.direction, cam.view);
        glm::vec3 pFocus = segment.ray.origin + ft * segment.ray.direction;

		segment.ray.origin += pLens;
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
	int* triangles_idx,
    ShadeableIntersection* intersections,
    BVHNode* bvh_nodes,
    Texture* textures,
    cudaTextureObject_t* texture_objects)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv(-1.f);
        glm::vec3 tangent;
        glm::vec3 bitangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
        glm::vec3 tmp_tangent;
        glm::vec3 tmp_bitangent;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH)
            {
#if BVH

				t = BVHIntersectionTest(geom, triangles, triangles_idx, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, tmp_bitangent, outside, bvh_nodes);
#else
                t = meshIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, tmp_bitangent, outside);
#endif
			}

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
				uv = tmp_uv;
                tangent = tmp_tangent;
                bitangent = tmp_bitangent;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
            intersections[path_index].textureId = geoms[hit_geom_index].textureIndex;
            intersections[path_index].tangent = tangent;
            intersections[path_index].bitangent = bitangent;
			intersections[path_index].normalId = geoms[hit_geom_index].normalIndex;

        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* texture_objects,
    cudaTextureObject_t* normal_objects,
    cudaTextureObject_t* env_objects)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (pathSegments[idx].remainingBounces <= 0) {
                //pathSegments[idx].color = glm::vec3(0.0f);
                return;
            }

            if (material.emittance > 0.0f) {
                // material is a light
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                //bump map
                if (intersection.normalId != -1) {
                    cudaTextureObject_t normal_obj = normal_objects[intersection.normalId];
                    float4 uv_normal = tex2D<float4>(normal_obj, intersection.uv.x, intersection.uv.y);
					glm::vec3 tan_normal = glm::vec3(uv_normal.x, uv_normal.y, uv_normal.z);
					tan_normal = glm::normalize(tan_normal * 2.0f - 1.0f);
					glm::mat3 TBN = glm::mat3(intersection.tangent, intersection.bitangent, intersection.surfaceNormal);
					intersection.surfaceNormal = glm::normalize(TBN * tan_normal);
                }

                pathSegments[idx].remainingBounces--;
                glm::vec3 intersect_pt = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], intersect_pt, intersection.surfaceNormal, material, rng);

                // texturing
                if (intersection.textureId != -1) {
					cudaTextureObject_t tex_obj = texture_objects[intersection.textureId];
					float4 uv_color = tex2D<float4>(tex_obj, intersection.uv.x, intersection.uv.y);
					glm::vec3 texture_color = glm::vec3(uv_color.x, uv_color.y, uv_color.z);
                    pathSegments[idx].color *= texture_color;
                }

                if (material.isProcedural) {
                    // checkerboard procedural texture
                    float scale_factor = material.checker_scale;
                    glm::vec3 color1 = material.checker_color1;
                    glm::vec3 color2 = material.checker_color2;
                    glm::vec2 uv = intersection.uv * scale_factor;
                    int u = (int)floor(uv.x);
                    int v = (int)floor(uv.y);
                    bool even = ((u + v) % 2 == 0);
                    glm::vec3 check_color = even ? color1 : color2;
                    pathSegments[idx].color *= check_color;
                }

                else {
                    pathSegments[idx].color *= material.color;
                }

            }
        }
        else {
            // no intersection -> environment map
            if (env_objects != NULL) {
				glm::vec3 env_dir = glm::normalize(pathSegments[idx].ray.direction);
				float u = 0.5f + (atan2(env_dir.z, env_dir.x) / (2.0f * PI));
				float v = 0.5f - (asin(env_dir.y) / PI);
				cudaTextureObject_t env_obj = env_objects[0];
				float4 uv_color = tex2D<float4>(env_obj, u, v);
                glm::vec3 env_color = glm::vec3(uv_color.x, uv_color.y, uv_color.z);
                pathSegments[idx].color *= env_color;
            }
            else {
                pathSegments[idx].color *= glm::vec3(0.0f);
            }
			pathSegments[idx].remainingBounces = 0;
            //pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct isPathDone {
    __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces != 0;
    }
};

struct materialComp {
    __host__ __device__
        bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) const {
        return m1.materialId < m2.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int og_num_paths = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_triangles_idx,
            dev_intersections,
            dev_bvhNodes,
            dev_textures,
			dev_texture_objects
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIALS
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialComp());
#endif

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_texture_objects,
            dev_normal_objects,
            dev_env_objects
            );
        checkCUDAError("shader");
        cudaDeviceSynchronize();

#if STREAM_COMPACTION
        // stream compaction
        PathSegment* new_dev_path_end = thrust::stable_partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            isPathDone()
        );
        num_paths = new_dev_path_end - dev_paths;
        dev_path_end = new_dev_path_end;
#endif


        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (og_num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
