#include "dvs/raycast.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace dvs {
namespace {

constexpr float kRayEps = 1e-6f;
constexpr int kMaxStackSize = 128;

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
        }                                                                                      \
    } while (0)

__device__ bool intersect_aabb(const Ray& ray, const AABB& box, float t_limit) {
    float tmin = 0.0f;
    float tmax = t_limit;
    for (int axis = 0; axis < 3; ++axis) {
        const float o = ray.origin[axis];
        const float d = ray.direction[axis];
        const float bmin = box.min[axis];
        const float bmax = box.max[axis];

        if (fabsf(d) < 1e-8f) {
            if (o < bmin || o > bmax) {
                return false;
            }
            continue;
        }

        float t0 = (bmin - o) / d;
        float t1 = (bmax - o) / d;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax < tmin) {
            return false;
        }
    }
    return true;
}

__device__ bool intersect_triangle(
    const Ray& ray, const Triangle& tri, float* out_t, float* out_u, float* out_v) {
    const Vec3 edge1 = tri.v1 - tri.v0;
    const Vec3 edge2 = tri.v2 - tri.v0;
    const Vec3 pvec = cross(ray.direction, edge2);
    const float det = dot(edge1, pvec);
    if (fabsf(det) < kRayEps) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 tvec = ray.origin - tri.v0;
    const float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 qvec = cross(tvec, edge1);
    const float v = dot(ray.direction, qvec) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }
    const float t = dot(edge2, qvec) * inv_det;
    if (t <= kRayEps) {
        return false;
    }
    *out_t = t;
    *out_u = u;
    *out_v = v;
    return true;
}

__device__ inline std::uint32_t expand_bits_10(std::uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline std::uint32_t morton3d_30bit(float x, float y, float z) {
    x = clampf(x, 0.0f, 0.999999f);
    y = clampf(y, 0.0f, 0.999999f);
    z = clampf(z, 0.0f, 0.999999f);
    const auto xx = static_cast<std::uint32_t>(x * 1024.0f);
    const auto yy = static_cast<std::uint32_t>(y * 1024.0f);
    const auto zz = static_cast<std::uint32_t>(z * 1024.0f);
    return (expand_bits_10(xx) << 2) | (expand_bits_10(yy) << 1) | expand_bits_10(zz);
}

__global__ void compute_morton_codes_kernel(
    const Triangle* triangles,
    int tri_count,
    Vec3 centroid_min,
    Vec3 centroid_max,
    std::uint32_t* out_codes,
    int* out_indices) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tri_count) {
        return;
    }

    const Triangle tri = triangles[tid];
    const Vec3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    const Vec3 extent = centroid_max - centroid_min;
    const float nx = (centroid.x - centroid_min.x) / fmaxf(extent.x, 1e-9f);
    const float ny = (centroid.y - centroid_min.y) / fmaxf(extent.y, 1e-9f);
    const float nz = (centroid.z - centroid_min.z) / fmaxf(extent.z, 1e-9f);
    out_codes[tid] = morton3d_30bit(nx, ny, nz);
    out_indices[tid] = tid;
}

__global__ void reorder_triangles_kernel(
    const Triangle* src,
    const int* sorted_indices,
    int tri_count,
    Triangle* dst) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tri_count) {
        return;
    }
    dst[tid] = src[sorted_indices[tid]];
}

__device__ int common_prefix(const std::uint32_t* codes, int n, int i, int j) {
    if (j < 0 || j >= n) {
        return -1;
    }
    const std::uint32_t a = codes[i];
    const std::uint32_t b = codes[j];
    if (a == b) {
        return 32 + __clz(static_cast<unsigned int>(i ^ j));
    }
    return __clz(a ^ b);
}

__device__ int find_split(const std::uint32_t* codes, int first, int last) {
    const std::uint32_t first_code = codes[first];
    const std::uint32_t last_code = codes[last];

    if (first_code == last_code) {
        return (first + last) >> 1;
    }

    const int common_prefix_len = __clz(first_code ^ last_code);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int new_split = split + step;
        if (new_split < last) {
            const int split_prefix_len = __clz(first_code ^ codes[new_split]);
            if (split_prefix_len > common_prefix_len) {
                split = new_split;
            }
        }
    } while (step > 1);

    return split;
}

__global__ void init_leaf_nodes_kernel(const Triangle* triangles, int tri_count, BVHNode* nodes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tri_count) {
        return;
    }
    const int node_idx = tri_count - 1 + tid;
    const Triangle tri = triangles[tid];
    BVHNode node{};
    node.left = -1;
    node.right = -1;
    node.start = tid;
    node.count = 1;
    node.box.min = min_vec3(min_vec3(tri.v0, tri.v1), tri.v2);
    node.box.max = max_vec3(max_vec3(tri.v0, tri.v1), tri.v2);
    nodes[node_idx] = node;
}

__global__ void build_internal_nodes_kernel(
    const std::uint32_t* morton_codes, int tri_count, BVHNode* nodes, int* parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tri_count - 1) {
        return;
    }

    const int lcp_next = common_prefix(morton_codes, tri_count, i, i + 1);
    const int lcp_prev = common_prefix(morton_codes, tri_count, i, i - 1);
    const int dir = (lcp_next - lcp_prev) >= 0 ? 1 : -1;
    const int delta_min = common_prefix(morton_codes, tri_count, i, i - dir);

    int lmax = 2;
    while (common_prefix(morton_codes, tri_count, i, i + lmax * dir) > delta_min) {
        lmax <<= 1;
    }

    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1) {
        if (common_prefix(morton_codes, tri_count, i, i + (l + t) * dir) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * dir;
    const int first = (i < j) ? i : j;
    const int last = (i > j) ? i : j;

    const int split = find_split(morton_codes, first, last);
    const int left_child = (split == first) ? (tri_count - 1 + split) : split;
    const int right_child = (split + 1 == last) ? (tri_count - 1 + split + 1) : (split + 1);

    BVHNode node{};
    node.left = left_child;
    node.right = right_child;
    node.start = 0;
    node.count = 0;
    nodes[i] = node;

    parent[left_child] = i;
    parent[right_child] = i;
}

__global__ void propagate_bounds_from_leaves_kernel(
    int tri_count, BVHNode* nodes, const int* parent, int* ready_flags) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tri_count) {
        return;
    }

    int current = tri_count - 1 + tid;
    int p = parent[current];
    while (p != -1) {
        const int old = atomicAdd(&ready_flags[p], 1);
        if (old == 0) {
            return;
        }
        const BVHNode left = nodes[nodes[p].left];
        const BVHNode right = nodes[nodes[p].right];
        nodes[p].box.min = min_vec3(left.box.min, right.box.min);
        nodes[p].box.max = max_vec3(left.box.max, right.box.max);
        current = p;
        p = parent[current];
    }
}

__global__ void batch_intersect_kernel(
    const BVHNode* nodes,
    int node_count,
    int root_index,
    const Triangle* triangles,
    const Ray* rays,
    int ray_count,
    Intersection* out_hits) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ray_count) {
        return;
    }
    if (node_count <= 0) {
        out_hits[tid] = Intersection{};
        return;
    }

    const Ray ray = rays[tid];
    float closest_t = 1e30f;
    int closest_tri_idx = -1;
    float best_u = 0.0f;
    float best_v = 0.0f;

    int stack[kMaxStackSize];
    int stack_size = 0;
    stack[stack_size++] = root_index;

    while (stack_size > 0) {
        const int node_idx = stack[--stack_size];
        const BVHNode node = nodes[node_idx];
        if (!intersect_aabb(ray, node.box, closest_t)) {
            continue;
        }

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int tri_idx = node.start + i;
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (!intersect_triangle(ray, triangles[tri_idx], &t, &u, &v)) {
                    continue;
                }
                if (t < closest_t) {
                    closest_t = t;
                    closest_tri_idx = tri_idx;
                    best_u = u;
                    best_v = v;
                }
            }
            continue;
        }

        if (node.left >= 0 && stack_size < kMaxStackSize) {
            stack[stack_size++] = node.left;
        }
        if (node.right >= 0 && stack_size < kMaxStackSize) {
            stack[stack_size++] = node.right;
        }
    }

    if (closest_tri_idx < 0) {
        out_hits[tid] = Intersection{};
        return;
    }

    const Triangle tri = triangles[closest_tri_idx];
    const float w = 1.0f - best_u - best_v;
    const Vec3 interp_n = normalize(tri.n0 * w + tri.n1 * best_u + tri.n2 * best_v);
    const Vec3 hit_point = ray.origin + ray.direction * closest_t;

    Intersection out;
    out.hit = 1;
    out.t = closest_t;
    out.triangle_index = closest_tri_idx;
    out.point = hit_point;
    out.normal = interp_n;
    out_hits[tid] = out;
}

}  // namespace

struct GpuRaycaster::Impl {
    int node_count = 0;
    int tri_count = 0;
    int root_index = 0;
    BVHNode* d_nodes = nullptr;
    Triangle* d_tris = nullptr;
    mutable Ray* d_rays = nullptr;
    mutable Intersection* d_hits = nullptr;
    mutable std::size_t ray_capacity = 0;

    void release() {
        if (d_nodes != nullptr) {
            cudaFree(d_nodes);
            d_nodes = nullptr;
        }
        if (d_tris != nullptr) {
            cudaFree(d_tris);
            d_tris = nullptr;
        }
        if (d_rays != nullptr) {
            cudaFree(d_rays);
            d_rays = nullptr;
        }
        if (d_hits != nullptr) {
            cudaFree(d_hits);
            d_hits = nullptr;
        }
        node_count = 0;
        tri_count = 0;
        root_index = 0;
        ray_capacity = 0;
    }

    void init_with_prebuilt_bvh(
        const std::vector<BVHNode>& nodes, const std::vector<Triangle>& triangles) {
        node_count = static_cast<int>(nodes.size());
        tri_count = static_cast<int>(triangles.size());
        if (node_count <= 0 || tri_count <= 0) {
            throw std::invalid_argument("prebuilt BVH requires non-empty nodes and triangles");
        }

        root_index = 0;
        CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode) * static_cast<std::size_t>(node_count)));
        CUDA_CHECK(cudaMalloc(&d_tris, sizeof(Triangle) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMemcpy(
            d_nodes,
            nodes.data(),
            sizeof(BVHNode) * static_cast<std::size_t>(node_count),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_tris,
            triangles.data(),
            sizeof(Triangle) * static_cast<std::size_t>(tri_count),
            cudaMemcpyHostToDevice));
    }

    void init_with_gpu_lbvh(const std::vector<Triangle>& triangles) {
        tri_count = static_cast<int>(triangles.size());
        if (tri_count <= 0) {
            throw std::invalid_argument("triangles must be non-empty");
        }

        Triangle* d_input_tris = nullptr;
        std::uint32_t* d_codes = nullptr;
        int* d_sorted_indices = nullptr;
        int* d_parent = nullptr;
        int* d_ready = nullptr;

        CUDA_CHECK(cudaMalloc(&d_tris, sizeof(Triangle) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMalloc(&d_input_tris, sizeof(Triangle) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMemcpy(
            d_input_tris,
            triangles.data(),
            sizeof(Triangle) * static_cast<std::size_t>(tri_count),
            cudaMemcpyHostToDevice));

        if (tri_count == 1) {
            node_count = 1;
            root_index = 0;
            CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode)));
            const int threads = 256;
            const int blocks = 1;
            init_leaf_nodes_kernel<<<blocks, threads>>>(d_input_tris, tri_count, d_nodes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_input_tris));
            return;
        }

        node_count = tri_count * 2 - 1;
        CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode) * static_cast<std::size_t>(node_count)));
        CUDA_CHECK(cudaMalloc(&d_codes, sizeof(std::uint32_t) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMalloc(&d_sorted_indices, sizeof(int) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * static_cast<std::size_t>(node_count)));
        CUDA_CHECK(cudaMalloc(&d_ready, sizeof(int) * static_cast<std::size_t>(tri_count - 1)));

        Vec3 centroid_min{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        Vec3 centroid_max{
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        for (const Triangle& tri : triangles) {
            const Vec3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            centroid_min = min_vec3(centroid_min, centroid);
            centroid_max = max_vec3(centroid_max, centroid);
        }

        const int threads = 256;
        const int tri_blocks = (tri_count + threads - 1) / threads;
        const int internal_blocks = ((tri_count - 1) + threads - 1) / threads;

        compute_morton_codes_kernel<<<tri_blocks, threads>>>(
            d_input_tris, tri_count, centroid_min, centroid_max, d_codes, d_sorted_indices);
        CUDA_CHECK(cudaGetLastError());

        thrust::sort_by_key(
            thrust::device,
            thrust::device_pointer_cast(d_codes),
            thrust::device_pointer_cast(d_codes + tri_count),
            thrust::device_pointer_cast(d_sorted_indices));

        reorder_triangles_kernel<<<tri_blocks, threads>>>(
            d_input_tris, d_sorted_indices, tri_count, d_tris);
        CUDA_CHECK(cudaGetLastError());

        init_leaf_nodes_kernel<<<tri_blocks, threads>>>(d_tris, tri_count, d_nodes);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_parent, 0xFF, sizeof(int) * static_cast<std::size_t>(node_count)));
        build_internal_nodes_kernel<<<internal_blocks, threads>>>(d_codes, tri_count, d_nodes, d_parent);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_ready, 0, sizeof(int) * static_cast<std::size_t>(tri_count - 1)));
        propagate_bounds_from_leaves_kernel<<<tri_blocks, threads>>>(
            tri_count, d_nodes, d_parent, d_ready);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> h_parent(static_cast<std::size_t>(node_count), -1);
        CUDA_CHECK(cudaMemcpy(
            h_parent.data(),
            d_parent,
            sizeof(int) * static_cast<std::size_t>(node_count),
            cudaMemcpyDeviceToHost));

        int detected_root = -1;
        for (int i = 0; i < node_count; ++i) {
            if (h_parent[static_cast<std::size_t>(i)] == -1) {
                detected_root = i;
                break;
            }
        }
        if (detected_root < 0) {
            throw std::runtime_error("failed to determine LBVH root");
        }
        root_index = detected_root;

        CUDA_CHECK(cudaFree(d_input_tris));
        CUDA_CHECK(cudaFree(d_codes));
        CUDA_CHECK(cudaFree(d_sorted_indices));
        CUDA_CHECK(cudaFree(d_parent));
        CUDA_CHECK(cudaFree(d_ready));
    }

    Impl(const std::vector<BVHNode>& nodes, const std::vector<Triangle>& triangles) {
        try {
            if (triangles.empty()) {
                throw std::invalid_argument("triangles must be non-empty");
            }
            if (!nodes.empty()) {
                init_with_prebuilt_bvh(nodes, triangles);
            } else {
                init_with_gpu_lbvh(triangles);
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() {
        release();
    }

    void ensure_ray_capacity(std::size_t ray_count) const {
        if (ray_capacity >= ray_count) {
            return;
        }
        if (d_rays != nullptr) {
            CUDA_CHECK(cudaFree(d_rays));
            d_rays = nullptr;
        }
        if (d_hits != nullptr) {
            CUDA_CHECK(cudaFree(d_hits));
            d_hits = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&d_rays, sizeof(Ray) * ray_count));
        CUDA_CHECK(cudaMalloc(&d_hits, sizeof(Intersection) * ray_count));
        ray_capacity = ray_count;
    }

    std::vector<Intersection> intersect(const std::vector<Ray>& rays) const {
        if (rays.empty()) {
            return {};
        }
        ensure_ray_capacity(rays.size());
        CUDA_CHECK(cudaMemcpy(
            d_rays, rays.data(), sizeof(Ray) * rays.size(), cudaMemcpyHostToDevice));

        const int ray_count = static_cast<int>(rays.size());
        const int threads = 256;
        const int blocks = (ray_count + threads - 1) / threads;
        batch_intersect_kernel<<<blocks, threads>>>(
            d_nodes, node_count, root_index, d_tris, d_rays, ray_count, d_hits);
        CUDA_CHECK(cudaGetLastError());

        std::vector<Intersection> hits(rays.size());
        CUDA_CHECK(cudaMemcpy(
            hits.data(), d_hits, sizeof(Intersection) * hits.size(), cudaMemcpyDeviceToHost));
        return hits;
    }
};

GpuRaycaster::GpuRaycaster(
    const std::vector<BVHNode>& nodes, const std::vector<Triangle>& triangles)
    : impl_(std::make_unique<Impl>(nodes, triangles)) {}

GpuRaycaster::~GpuRaycaster() = default;

GpuRaycaster::GpuRaycaster(GpuRaycaster&& other) noexcept = default;

GpuRaycaster& GpuRaycaster::operator=(GpuRaycaster&& other) noexcept = default;

std::vector<Intersection> GpuRaycaster::intersect(const std::vector<Ray>& rays) const {
    return impl_->intersect(rays);
}

}  // namespace dvs
