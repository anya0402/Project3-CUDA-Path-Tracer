#include "scene.h"

#include "utilities.h"
#include "tiny_obj_loader.h"
#include "bvh.h"
#include "stb_image.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

// code adapated from tinyobjloader example
void Scene::loadFromObj(const std::string& pathName, Geom& mesh)
{
    std::string inputfile = pathName;
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

	int num_triangles = 0;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            Triangle new_tri;
			num_triangles++;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				new_tri.vertices[v] = glm::vec3(vx, vy, vz);
                new_tri.centroid = (glm::vec3(vx) + glm::vec3(vy) + glm::vec3(vz)) * 0.3333f;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    new_tri.normals[v] = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

					new_tri.uvs[v] = glm::vec2(tx, ty);
                }

                

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            int new_materialid = shapes[s].mesh.material_ids[f];
            new_tri.materialid = new_materialid >= 0 ? new_materialid : -1;

            // tangents and bitangents
            glm::vec3 edge1 = new_tri.vertices[1] - new_tri.vertices[0];
            glm::vec3 edge2 = new_tri.vertices[2] - new_tri.vertices[0];
            glm::vec2 deltaUV1 = new_tri.uvs[1] - new_tri.uvs[0];
            glm::vec2 deltaUV2 = new_tri.uvs[2] - new_tri.uvs[0];
            float tan_f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            glm::vec3 tangent;
            glm::vec3 bitangent;
            tangent.x = tan_f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = tan_f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = tan_f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            bitangent.x = tan_f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
            bitangent.y = tan_f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
            bitangent.z = tan_f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
            new_tri.tangent = glm::normalize(tangent);
            new_tri.bitangent = glm::normalize(bitangent);

            triangles.push_back(new_tri);
        }
    }
	mesh.numTriangles = num_triangles;
	mesh.meshStartIdx = triangles.size() - num_triangles;
	mesh.meshEndIdx = triangles.size() - 1;

    //bvh creation
    BVH bvh_instance(triangles);
    bvh_instance.bvhNodes = new BVHNode[triangles.size() * 2];
    bvh_instance.constructBVH();

	int root_offset = bvhNodes.size();
    mesh.bvhRootIdx = root_offset;

	mesh.numNodes = bvh_instance.nodes_used;

	

    for (int i = 0; i < bvh_instance.nodes_used; ++i) {
		BVHNode node_to_add = bvh_instance.bvhNodes[i];
        if (node_to_add.leftChild != -1) {
			node_to_add.leftChild += root_offset;
        }
		if (node_to_add.rightChild != -1) {
            node_to_add.rightChild += root_offset;
		}
		bvhNodes.push_back(node_to_add);
    }

	triangles_idx.insert(triangles_idx.end(), bvh_instance.triangles_idx.begin(), bvh_instance.triangles_idx.end());

    delete[] bvh_instance.bvhNodes;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Reflect")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.f;
        }
        else if (p["TYPE"] == "Refract")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasRefractive = 1.f;
			newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasReflective = 1.f;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "mesh")
        {
            newGeom.type = MESH;

            std::string path_for_obj;
            auto source_path = jsonName.substr(0, jsonName.find_last_of('/'));
			path_for_obj = source_path + "/" + std::string(p["OBJFILE"]) + ".obj";

			loadFromObj(path_for_obj, newGeom);

            //texture
            if (p.contains("TEXTURE")) {
                int texId = loadTexture(path_for_obj, std::string(p["TEXTURE"]));
                if (texId != -1) {
                    newGeom.textureIndex = texId;
                    newGeom.hasTexture = true;
                }
                else {
                    newGeom.hasTexture = false;
                }
            }
            else {
                newGeom.hasTexture = false;
			}

            // bump map
            if (p.contains("BUMP")) {
                int normalId = loadBumpMap(path_for_obj, std::string(p["BUMP"]));
                if (normalId != -1) {
                    newGeom.normalIndex = normalId;
                    newGeom.hasNormal = true;
                }
                else {
                    newGeom.hasNormal = false;
                }
            }
            else {
                newGeom.hasNormal = false;
            }


        }
        else {
			newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //depth of field
	camera.lensRadius = cameraData["RADIUS"];
	camera.focalDistance = cameraData["FOCALDIST"];

    //environment map
    if (data.contains("Environment")) {
		const auto& env_data = data["Environment"];
        std::string path_for_env;
        auto source_path_env = jsonName.substr(0, jsonName.find_last_of('/'));
        path_for_env = source_path_env + "/" + std::string(env_data["ENVFILE"]);

        int envId = loadEnvMap(path_for_env);
        if (envId == -1) {
            std::cout << "Failed to load environment map: " << std::string(data["Environment"]) << std::endl;
        }
	}

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

int Scene::loadTexture(std::string pathName, std::string textName) {
    int width, height, channels;
    const std::size_t lastSlashPos{ pathName.find_last_of('/') };
    pathName = pathName.substr(0, lastSlashPos) + std::string("/") + textName;

    unsigned char* data = stbi_load(pathName.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        std::cerr << "Failed to load texture: " << pathName << std::endl;
        return -1;
    }

    channels = 4;
    Texture texture;
    texture.width = width;
    texture.height = height;
    texture.channels = channels;
    texture.data = data;

    int textureId = textures.size();
    textures.push_back(texture);

    return textureId;
}

int Scene::loadBumpMap(const std::string pathName, const std::string textName) {
    int width, height, channels;
    const std::size_t lastSlashPos{ pathName.find_last_of('/') };
    std::string fullPath = pathName.substr(0, lastSlashPos) + std::string("/") + textName;

    unsigned char* data = stbi_load(fullPath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        std::cerr << "Failed to load bump map: " << fullPath << std::endl;
        return -1;
    }

    //channels = 4;
    Texture normal;
    normal.width = width;
    normal.height = height;
    normal.channels = channels;
    normal.data = data;

    int normalId = normals.size();
    normals.push_back(normal);

    return normalId;
}

int Scene::loadEnvMap(const std::string pathName) {
    int width, height, channels;
    stbi_set_flip_vertically_on_load(true);

   /* const std::size_t lastSlashPos{ pathName.find_last_of('/') };
    std::string fullPath = pathName.substr(0, lastSlashPos) + std::string("/") + textName;*/

    unsigned char* data = stbi_load(pathName.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    //float* data = stbi_loadf(pathName.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load environment map: " << pathName << std::endl;
        return -1;
    }

    //channels = 4;
    Texture env_map;
    env_map.width = width;
    env_map.height = height;
    env_map.channels = channels;
    env_map.data = data;

    int envId = env_maps.size();
    env_maps.push_back(env_map);

    return envId;
}