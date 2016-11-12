#pragma once

#define DEBUG_MODE 0
#if DEBUG_MODE
#include <stdio.h>
#define printf_DEBUG(...) printf( __VA_ARGS__ )
#else
#define printf_DEBUG(...) /*byte me*/
#endif

#define DEBUG_BVH 0
#if DEBUG_BVH
#include <stdio.h>
#define printf_DEBUG1(...) printf( __VA_ARGS__ )
#else
#define printf_DEBUG1(...) /*byte me*/
#endif

