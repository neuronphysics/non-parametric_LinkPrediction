//
// Created by su999 on 2023/3/13.
//
#ifndef OUTPUT_LEVEL
#define OUTPUT_LEVEL 3
#endif

#ifndef GLFM_CUDA_ACC_LOG_H
#define GLFM_CUDA_ACC_LOG_H

#define OUTPUT_CONCURRENT 4
#define OUTPUT_DEBUG 3
#define OUTPUT_INFO 2
#define OUTPUT_NORMAL 1

#define VERSION_DECLARE "Test old 800 with Q update \n"

#define LOG(level, fmt, ...) \
        if(OUTPUT_LEVEL >= level){ \
            printf(fmt, ##__VA_ARGS__);                       \
            printf("\n");         \
        }                     \

#endif //GLFM_CUDA_ACC_LOG_H
