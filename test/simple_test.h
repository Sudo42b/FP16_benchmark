#ifndef SIMPLE_TEST_H
#define SIMPLE_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT_TRUE(condition, message) \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s at %s:%d\n", message.c_str(), __FILE__, __LINE__); \
            exit(1); \
        }

#define ASSERT_GT(expected, actual, message) \
        if ((expected) <= (actual)) { \
            fprintf(stderr, "FAIL: %s - Expected %g, got %g at %s:%d\n", \
                    message.c_str(), (double)(expected), (double)(actual), __FILE__, __LINE__); \
            exit(1); \
        }
        
#define ASSERT_EQ(expected, actual, message) \
        if ((expected) != (actual)) { \
            fprintf(stderr, "FAIL: %s - Expected %g, got %g at %s:%d\n", \
                    message.c_str(), (double)(expected), (double)(actual), __FILE__, __LINE__); \
            exit(1); \
        }

#define EXPECT_EQ(expected, actual, message) \
        if ((expected) != (actual)) { \
            fprintf(stderr, "FAIL: %s - Expected %g, got %g at %s:%d\n", \
                    message.c_str(), (double)(expected), (double)(actual), __FILE__, __LINE__); \
        }

#define RUN_TEST(test_func) \
    printf("Running %s...\n", #test_func); \
    test_func(); \
    printf("PASS: %s\n", #test_func);

#endif // SIMPLE_TEST_H 