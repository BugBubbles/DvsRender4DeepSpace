#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

#ifdef __CUDACC__
#define DVS_HD __host__ __device__
#else
#define DVS_HD
#endif

namespace dvs {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    DVS_HD float operator[](int axis) const {
        if (axis == 0) {
            return x;
        }
        if (axis == 1) {
            return y;
        }
        return z;
    }
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct Mat4 {
    float m[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

DVS_HD inline Vec3 make_vec3(float x, float y, float z) {
    return Vec3{x, y, z};
}

DVS_HD inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

DVS_HD inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

DVS_HD inline Vec3 operator-(const Vec3& v) {
    return Vec3{-v.x, -v.y, -v.z};
}

DVS_HD inline Vec3 operator*(const Vec3& v, float s) {
    return Vec3{v.x * s, v.y * s, v.z * s};
}

DVS_HD inline Vec3 operator*(float s, const Vec3& v) {
    return v * s;
}

DVS_HD inline Vec3 operator/(const Vec3& v, float s) {
    return Vec3{v.x / s, v.y / s, v.z / s};
}

#ifndef CUDACC

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
return os;
}

inline std::ostream& operator<<(std::ostream& os, const Vec4& v) {
os << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
return os;
}

inline std::ostream& operator<<(std::ostream& os, const Mat4& m) {
os << "[";
for (int i = 0; i < 4; ++i) {
    os << "[";
    for (int j = 0; j < 4; ++j) {
        os << m.m[i * 4 + j];
        if (j < 3) {
            os << ", ";
        }
    }
    os << "]";
    if (i < 3) {
        os << ",\n";
    }
}
os << "]";
return os;
}
#endif

DVS_HD inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

DVS_HD inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

DVS_HD inline float length(const Vec3& v) {
    return sqrtf(dot(v, v));
}

DVS_HD inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len < 1e-12f) {
        return make_vec3(0.0f, 0.0f, 0.0f);
    }
    return v / len;
}

DVS_HD inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

DVS_HD inline Vec3 min_vec3(const Vec3& a, const Vec3& b) {
    return Vec3{
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    };
}

DVS_HD inline Vec3 max_vec3(const Vec3& a, const Vec3& b) {
    return Vec3{
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    };
}

DVS_HD inline Vec4 mat4_mul_vec4(const Mat4& m, const Vec4& v) {
    return Vec4{
        m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z + m.m[3] * v.w,
        m.m[4] * v.x + m.m[5] * v.y + m.m[6] * v.z + m.m[7] * v.w,
        m.m[8] * v.x + m.m[9] * v.y + m.m[10] * v.z + m.m[11] * v.w,
        m.m[12] * v.x + m.m[13] * v.y + m.m[14] * v.z + m.m[15] * v.w
    };
}

}  // namespace dvs
