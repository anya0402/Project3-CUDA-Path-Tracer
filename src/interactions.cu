#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>


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

__host__ __device__ glm::vec3 reflectRay(glm::vec3 &wi, glm::vec3 &normal) {
    // from PBR Chapter 9.3
	return wi - 2.0f * glm::dot(wi, normal) * normal;
}

__host__ __device__ glm::vec3 refractRay(glm::vec3 &wi, glm::vec3 &normal, glm::vec3 &wt, float eta) {
	// from PBR Chapter 9.3
	float cosTheta_val = glm::dot(wi, normal);
    if (cosTheta_val < 0) {
        eta = 1.0f / eta;
        cosTheta_val = -cosTheta_val;
        normal = -normal;
    }
    float sin2Theta_val = std::max<float>(0, 1.0f - cosTheta_val * cosTheta_val);
	float sin2Theta_t = sin2Theta_val / (eta * eta);
    if (sin2Theta_t >= 1.0f) {
        return glm::vec3(0.0f); // total internal reflection case
    }
	float cosTheta_t = sqrt(1.0f - sin2Theta_t);
	wt = (-wi / eta) + (cosTheta_val / eta - cosTheta_t) * normal;
	return wt;
}

__host__ __device__ float fresnelRay(float cosTheta_val, float eta) {
    // from PBR Chapter 9.3
	cosTheta_val = glm::clamp(cosTheta_val, -1.0f, 1.0f);
    if (cosTheta_val < 0) {
        eta = 1.0f / eta;
        cosTheta_val = -cosTheta_val;
    }
    float sin2Theta_val = std::max<float>(0, 1.0f - cosTheta_val * cosTheta_val);
    float sin2Theta_t = sin2Theta_val / (eta * eta);
    if (sin2Theta_t >= 1.0f) {
        return 1.f; // total internal reflection case
    }
    float cosTheta_t = sqrt(1.0f - sin2Theta_t);

    float r_parl = ((eta * cosTheta_val) - cosTheta_t) / ((eta * cosTheta_val) + cosTheta_t);
	float r_perp = (cosTheta_val - (eta * cosTheta_t)) / (cosTheta_val + (eta * cosTheta_t));
    return (r_parl * r_parl + r_perp * r_perp) / 2.0f;
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
    glm::vec3 new_color = m.color;

    //if (m.hasReflective && !(m.hasRefractive)) {
    if (m.hasReflective) {
        // mirror
		new_dir = reflectRay(wi, normal);
		new_color = m.color;
    }
  //  else if (m.hasRefractive) {
  //      // glass
	//	//new_color = m.color;
	//	//new_dir = glm::vec3(0.0f);
 // //      // ***FINISH***

  //  }
    else {
        // diffuse
		new_dir = calculateRandomDirectionInHemisphere(normal, rng);
        new_color = m.color;
    }

    pathSegment.color *= new_color;
    pathSegment.ray.direction = glm::normalize(new_dir);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * .0001f;
    pathSegment.remainingBounces--;

    // original diffuse
	//glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
	//pathSegment.ray.origin = intersect;
	//pathSegment.color *= m.color;
	//pathSegment.ray.direction = wi;
	//pathSegment.remainingBounces--;

}
