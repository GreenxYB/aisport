/**
 * @author mpj
 * @date 2025/10/25 00:04
 * @version V1.0
 */

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

#ifndef uint8_t
typedef unsigned char uint8_t;
#endif

// ============ Device Functions ============

__device__ void affine_project(const float* matrix, float x, float y, float* out_x, float* out_y) {
    *out_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *out_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__device__ float box_iou(float aleft, float atop, float aright, float abottom,
                         float bleft, float btop, float bright, float bbottom) {
    float cleft = fmaxf(aleft, bleft);
    float ctop = fmaxf(atop, btop);
    float cright = fminf(aright, bright);
    float cbottom = fminf(abottom, bbottom);

    float c_area = fmaxf(cright - cleft, 0.0f) * fmaxf(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = fmaxf(0.0f, aright - aleft) * fmaxf(0.0f, abottom - atop);
    float b_area = fmaxf(0.0f, bright - bleft) * fmaxf(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

// ============ Kernels ============

extern "C" __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    unsigned char* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height, unsigned char const_value,
    float* matrix_2_3) {
        unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;
        // cuda的线程数可能会超过图像的大小，所以需要判断
        if (dx >= dst_width || dy >= dst_height) return;

        // 放射变换逆矩阵的值
        float m_x1 = matrix_2_3[0];
        float m_y1 = matrix_2_3[1];
        float m_z1 = matrix_2_3[2];
        float m_x2 = matrix_2_3[3];
        float m_y2 = matrix_2_3[4];
        float m_z2 = matrix_2_3[5];

        // 计算出原图的坐标
        float src_x = m_x1 * dx + m_y1 * dy + m_z1;
        float src_y = m_x2 * dx + m_y2 * dy + m_z2;

        float c0, c1, c2;

        if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
            // 如果超出原图的范围，就用const_value填充
            c0 = const_value;
            c1 = const_value;
            c2 = const_value;
        } else {
            int y_low = floorf(src_y);
            int x_low = floorf(src_x);
            int y_high = y_low + 1;
            int x_high = x_low + 1;

            uint8_t const_value_array[] = {const_value, const_value, const_value};
            float ly = src_y - y_low;
            float lx = src_x - x_low;
            float hy = 1 - ly;
            float hx = 1 - lx;
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            uint8_t *v1 = const_value_array;
            uint8_t *v2 = const_value_array;
            uint8_t *v3 = const_value_array;
            uint8_t *v4 = const_value_array;

            if (y_low >= 0) {
                if (x_low >= 0) v1 = (uint8_t *) (src + y_low * src_line_size + x_low * 3);
                if (x_high < src_width) v2 = (uint8_t *) (src + y_low * src_line_size + x_high * 3);
            }

            if (y_high < src_height) {
                if (x_low >= 0) v3 = (uint8_t *) (src + y_high * src_line_size + x_low * 3);
                if (x_high < src_width) v4 = (uint8_t *) (src + y_high * src_line_size + x_high * 3);
            }

            // same to opencv
            c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
            c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
            c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
        }

        // 进行swap RB操作，将BGR转换为RGB
        float tmp = c2;
        c2 = c0;
        c0 = tmp;

        c0 = c0 / 255.0f;
        c1 = c1 / 255.0f;
        c2 = c2 / 255.0f;

        int area = dst_width * dst_height;
        float *p_dst_c0 = dst + dy * dst_width + dx;
        float *p_dst_c1 = p_dst_c0 + area;
        float *p_dst_c2 = p_dst_c1 + area;
        *p_dst_c0 = c0;
        *p_dst_c1 = c1;
        *p_dst_c2 = c2;
}

extern "C" __global__ void decode_kernel_Pose(
    float *predict, int num_bboxes, int num_classes,
    int output_c_dim, float confidence_threshold,
    float *invert_affine_matrix, float *parray, const int max_image_boxes,
    int num_key_points, int num_pose_element) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_c_dim * position;
    float *class_confidence = pitem + 4;
    int class_label = 0;
    float confidence = *class_confidence++;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            class_label = i;
            confidence = *class_confidence;
        }
    }

    if (confidence < confidence_threshold)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;

    int index = atomicAdd(parray, 1);
    if (index >= max_image_boxes)
        return;

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * num_pose_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = class_label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore

    pitem += num_classes;
    for (int i = 0; i < num_key_points; ++i) {
        float keypoint_x = *pitem++;
        float keypoint_y = *pitem++;
        float keypoint_confidence = *pitem++;

        affine_project(invert_affine_matrix, keypoint_x, keypoint_y, &keypoint_x, &keypoint_y);

        *pout_item++ = keypoint_x;
        *pout_item++ = keypoint_y;
        *pout_item++ = keypoint_confidence;
    }
}

extern "C" __global__ void nms_kernel_Pose(float *bboxes, int max_objects, float threshold, int num_pose_element) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int) *bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag, (keypoint_x, keypoint_y, keypoint_confidence) * 17
    float *pcurrent = bboxes + 1 + position * num_pose_element;
    for (int i = 0; i < count; ++i) {
        float *pitem = bboxes + 1 + i * num_pose_element;
        if (i == position) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0], pitem[1], pitem[2], pitem[3]
            );

            if (iou > threshold) {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}
