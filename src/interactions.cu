#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <glm/glm.hpp>


__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 worldToLocalCoords(glm::vec3 normal, glm::vec3 wo) {
    // from PBR
    glm::vec3 w = glm::normalize(normal);
    glm::vec3 inter;
    if (fabs(w.x) > 1.0f) {
        inter = glm::vec3(0, 1, 0);
    }
    else {
		inter = glm::vec3(1, 0, 0);
    }
	glm::vec3 u = glm::normalize(glm::cross(inter, w));
    glm::vec3 v = glm::cross(w, u);
	return glm::vec3(glm::dot(wo, u), glm::dot(wo, v), glm::dot(wo, w));
}

__host__ __device__ glm::vec3 localToWorldCoords(glm::vec3 normal, glm::vec3 wo) {
    glm::vec3 w = glm::normalize(normal);
    glm::vec3 inter;
    if (fabs(w.x) > 1.0f) {
        inter = glm::vec3(0, 1, 0);
    }
    else {
        inter = glm::vec3(1, 0, 0);
    }
    glm::vec3 u = glm::normalize(glm::cross(inter, w));
    glm::vec3 v = glm::cross(w, u);
    return glm::vec3(glm::dot(wo, u), glm::dot(wo, v), glm::dot(wo, w)) * wo;
}

__host__ __device__ glm::vec3 reflectRay(glm::vec3 &wi, glm::vec3 &normal) {
    // from PBR Chapter 9.3
	return wi - 2.0f * glm::dot(wi, normal) * normal;
}

__host__ __device__ glm::vec3 refractRay(glm::vec3 &wi, glm::vec3 &normal, float eta) {
	// from PBR Chapter 9.3
	float cosTheta_val = glm::dot(wi, normal);
    //if (cosTheta_val < 0) {
    //    eta = 1.0f / eta;
    //    cosTheta_val = -cosTheta_val;
    //    normal = -normal;
    //}
    float sin2Theta_val = std::max<float>(0, 1.0f - cosTheta_val * cosTheta_val);
	float sin2Theta_t = sin2Theta_val * (eta * eta);
    if (sin2Theta_t >= 1.0f) {
        return glm::vec3(0.0f); // total internal reflection case
    }
	float cosTheta_t = sqrtf(1.0f - sin2Theta_t);
	glm::vec3 result = (-wi * eta) + (cosTheta_val * eta - cosTheta_t) * normal;
	return result;
}

__host__ __device__ float fresnelRay(float cosTheta_val, float eta_i, float eta_t) {
    // from PBR Chapter 9.3
    cosTheta_val = glm::clamp(cosTheta_val, -1.0f, 1.0f);
    bool entering = cosTheta_val <= 0;
    if (entering) {
        float temp = eta_i;
        eta_i = eta_t;
        eta_t = temp;
        cosTheta_val = abs(cosTheta_val);
    }
    float sin2Theta_val = fmaxf(0, 1.0f - cosTheta_val * cosTheta_val);
    float sin2Theta_t = eta_i / eta_t * sin2Theta_val;
    if (sin2Theta_t >= 1.0f) {
        return 1.f; // total internal reflection case
    }
    float cosTheta_t = sqrtf(fmaxf(0.0f, 1.0f - sin2Theta_t * sin2Theta_t));

    float r_parl = ((eta_t * cosTheta_val) - (eta_i - cosTheta_t)) / ((eta_t * cosTheta_val) + (eta_i - cosTheta_t));
    float r_perp = ((eta_i * cosTheta_val) - (eta_t - cosTheta_t)) / ((eta_i * cosTheta_val) + (eta_t - cosTheta_t));
    return (r_parl * r_parl + r_perp * r_perp) * 0.5f;

	//cosTheta_val = glm::clamp(cosTheta_val, -1.0f, 1.0f);
 //   if (cosTheta_val < 0) {
 //       eta = 1.0f / eta;
 //       cosTheta_val = -cosTheta_val;
 //   }
 //   float sin2Theta_val = fmaxf(0, 1.0f - cosTheta_val * cosTheta_val);
 //   float sin2Theta_t = sin2Theta_val / (eta * eta);
 //   if (sin2Theta_t >= 1.0f) {
 //       return 1.f; // total internal reflection case
 //   }
 //   float cosTheta_t = sqrtf(fmaxf(0.0f, 1.0f - sin2Theta_t));

	//float denom = (eta * cosTheta_val) + cosTheta_t;
	//if (denom == 0.0f) return 1.0f;

 //   float r_parl = ((eta * cosTheta_val) - cosTheta_t) / (denom);
	//float r_perp = (cosTheta_val - (eta * cosTheta_t)) / (denom);
 //   return (r_parl * r_parl + r_perp * r_perp) / 2.0f;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	glm::vec3 wi = glm::normalize(pathSegment.ray.direction);
    glm::vec3 new_dir(0.0f);
	normal = glm::normalize(normal);
    glm::vec3 new_color = m.color;
	bool reflect = m.hasReflective;
	bool refract = m.hasRefractive;

    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasReflective && !(m.hasRefractive)) {
        // mirror
		new_dir = reflectRay(wi, normal);

        pathSegment.ray.direction = glm::normalize(new_dir);
        pathSegment.color *= new_color;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * .0001f;
        //pathSegment.remainingBounces--;
    }


    else if (m.hasRefractive) {
        // glass
        float eta_i = m.indexOfRefraction;
        float eta_t = 1.f; // outer material (air)
        glm::vec3 w_local = -pathSegment.ray.direction;
        w_local = worldToLocalCoords(normal, w_local);
        float cosTheta_val = w_local.z;
        float fresnel_val = fresnelRay(cosTheta_val, eta_i, eta_t);
        float t = 1.0f - fresnel_val;

        if (u01(rng) < fresnel_val) {
            //reflect using fresnel
            new_dir = glm::vec3(-w_local.x, -w_local.y, w_local.z);
            t = fresnel_val;
			new_color = m.color * t / abs(new_dir.z);
        }
        else {
			//refract
            bool entering = cosTheta_val <= 0;
            if (entering) {
                float temp = eta_i;
                eta_i = eta_t;
                eta_t = temp;
            }
            float eta = eta_i / eta_t;

			glm::vec3 normal_local(0, 0, 1);
            if (glm::dot(normal_local, w_local) < 0) {
                normal_local = -normal_local;
            }

            new_dir = glm::refract(w_local, normal_local, eta);

            if (new_dir == glm::vec3(0.0f)) {
                pathSegment.remainingBounces = 0;
                return;
                //new_dir = reflectRay(wi, normal);
            }
            else {
                //new_dir = glm::normalize(new_dir);
                glm::vec3 ft = m.color * t;
				new_color = ft / abs(new_dir.z);
            }
        }
        glm::vec3 wi_world = localToWorldCoords(normal, new_dir);
        //glm::vec3 wi_world = new_dir;
        wi_world = glm::normalize(wi_world);

        new_color = new_color * glm::abs(glm::dot(wi_world, normal)) / t;

        // FIX THIS
        /*pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
        pathSegment.ray.origin += 0.01f * wi;*/  
        pathSegment.ray.direction = wi_world;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * .005f;
        pathSegment.color *= new_color;

        //pathSegment.remainingBounces--;

        /*float eta = m.indexOfRefraction;
		float cosTheta_val = glm::dot(wi, normal);
        if (cosTheta_val < 0) {
            eta = 1.0f / eta;
            cosTheta_val = -cosTheta_val;
            normal = -normal;
		}
		float fresnel_val = fresnelRay(cosTheta_val, eta);
		float t = 1.0f - fresnel_val;

        if (u01(rng) < fresnel_val) {
            new_dir = reflectRay(wi, normal);
        }
        else {
            new_dir = glm::refract(wi, normal, eta);
            if (new_dir == glm::vec3(0.0f)) {
                new_dir = reflectRay(wi, normal);
            }
            else {
                new_dir = glm::normalize(new_dir);
            }
        }*/

    }


    else {
        // diffuse
		new_dir = calculateRandomDirectionInHemisphere(normal, rng);

        pathSegment.ray.direction = glm::normalize(new_dir);
        pathSegment.color *= new_color;
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * .0001f;
        //pathSegment.remainingBounces--;
    }


    

    // original diffuse
	//glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
	//pathSegment.ray.origin = intersect;
	//pathSegment.color *= m.color;
	//pathSegment.ray.direction = wi;
	//pathSegment.remainingBounces--;

}
