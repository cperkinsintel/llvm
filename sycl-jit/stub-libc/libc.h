extern void __assert_fail(const char *, const char *, unsigned int,
                          const char *);

#define assert(expr)                                                           \
  (static_cast<bool>(expr)                                                     \
       ? void(0)                                                               \
       : __assert_fail(#expr, __FILE__, __LINE__, ((const char *)0)))

extern int errno;

typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
#ifndef _WIN32
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
#else
typedef signed long long int __int64_t;
typedef unsigned long long int __uint64_t;
#endif
typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;
typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;

typedef long int __intptr_t;

typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
typedef __int_least8_t int_least8_t;
typedef __int_least16_t int_least16_t;
typedef __int_least32_t int_least32_t;
typedef __int_least64_t int_least64_t;
typedef __uint_least8_t uint_least8_t;
typedef __uint_least16_t uint_least16_t;
typedef __uint_least32_t uint_least32_t;
typedef __uint_least64_t uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef long int intptr_t;
#ifndef _WIN32
typedef unsigned long int uintptr_t;
#else
typedef unsigned long long int uintptr_t;
#endif

typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;

typedef decltype(sizeof(void *)) size_t;

#define INT16_MAX (32767)
#define INT32_MAX (2147483647)
#define INT64_MAX (__INT64_C(9223372036854775807))
#define INT8_MAX (127)
#define INTMAX_C(c) c##L
#define INTMAX_MAX (__INT64_C(9223372036854775807))
#define INTMAX_MIN (-__INT64_C(9223372036854775807) - 1)
#define INTPTR_MAX (9223372036854775807L)
#define INT_FAST16_MAX (9223372036854775807L)
#define INT_FAST32_MAX (9223372036854775807L)
#define INT_FAST64_MAX (__INT64_C(9223372036854775807))
#define INT_FAST8_MAX (127)
#define INT_LEAST16_MAX (32767)
#define INT_LEAST32_MAX (2147483647)
#define INT_LEAST64_MAX (__INT64_C(9223372036854775807))
#define INT_LEAST8_MAX (127)
#define PTRDIFF_MAX (9223372036854775807L)
#define SIZE_MAX (18446744073709551615UL)
#define UINT16_MAX (65535)
#define UINT32_MAX (4294967295U)
#define UINT64_MAX (__UINT64_C(18446744073709551615))
#define UINT8_MAX (255)
#define UINTMAX_C(c) c##UL
#define UINTMAX_MAX (__UINT64_C(18446744073709551615))
#define UINTPTR_MAX (18446744073709551615UL)
#define UINT_FAST16_MAX (18446744073709551615UL)
#define UINT_FAST32_MAX (18446744073709551615UL)
#define UINT_FAST64_MAX (__UINT64_C(18446744073709551615))
#define UINT_FAST8_MAX (255)
#define UINT_LEAST16_MAX (65535)
#define UINT_LEAST32_MAX (4294967295U)
#define UINT_LEAST64_MAX (__UINT64_C(18446744073709551615))
#define UINT_LEAST8_MAX (255)
#define WCHAR_MAX __WCHAR_MAX
#define WINT_MAX (4294967295u)
#define INT16_MIN (-32767 - 1)
#define INT32_MIN (-2147483647 - 1)
#define INT64_MIN (-__INT64_C(9223372036854775807) - 1)
#define INT8_MIN (-128)
#define INTMAX_MIN (-__INT64_C(9223372036854775807) - 1)
#define INTPTR_MIN (-9223372036854775807L - 1)
#define INT_FAST16_MIN (-9223372036854775807L - 1)
#define INT_FAST32_MIN (-9223372036854775807L - 1)
#define INT_FAST64_MIN (-__INT64_C(9223372036854775807) - 1)
#define INT_FAST8_MIN (-128)
#define INT_LEAST16_MIN (-32767 - 1)
#define INT_LEAST32_MIN (-2147483647 - 1)
#define INT_LEAST64_MIN (-__INT64_C(9223372036854775807) - 1)
#define INT_LEAST8_MIN (-128)
#define PTRDIFF_MIN (-9223372036854775807L - 1)
#define WCHAR_MIN __WCHAR_MIN
#define WINT_MIN (0u)

#define FP_ILOGB0 (-2147483647 - 1)
#define FP_ILOGBNAN (-2147483647 - 1)
#define FP_INFINITE 1
#define FP_NAN 0
#define FP_NORMAL 4
#define FP_SUBNORMAL 3
#define FP_ZERO 2

typedef struct {
  int quot;
  int rem;
} div_t;
typedef struct {
  long int quot;
  long int rem;
} ldiv_t;
__extension__ typedef struct {
  long long int quot;
  long long int rem;
} lldiv_t;

extern div_t div(int, int);
extern ldiv_t ldiv(long int, long int);
extern lldiv_t lldiv(long long int, long long int);

extern void *malloc(size_t);
extern void free(void *);

extern "C" void *memcpy(void *, const void *, size_t);
extern int strcmp(const char *, const char *);

#define EOF (-1)

extern int remove(const char *);

typedef long int time_t;

struct timespec {
  long int tv_sec;
  long int tv_nsec;
};

extern int nanosleep(timespec *, timespec *);

typedef unsigned long int pthread_t;
typedef struct {
} pthread_mutex_t;
typedef struct {
} pthread_cond_t;
typedef struct {
} pthread_attr_t;
typedef struct {
} pthread_once_t;
typedef struct {
} pthread_key_t;
typedef struct {
} pthread_mutexattr_t;

extern int pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
extern int pthread_mutex_destroy(pthread_mutex_t *);
extern int pthread_mutex_trylock(pthread_mutex_t *);
extern int pthread_mutex_lock(pthread_mutex_t *);
extern int pthread_mutex_unlock(pthread_mutex_t *);

extern int pthread_mutexattr_init(pthread_mutexattr_t *);
extern int pthread_mutexattr_settype(pthread_mutexattr_t *, int);
extern int pthread_mutexattr_destroy(pthread_mutexattr_t *);

extern int pthread_cond_destroy(pthread_cond_t *);
extern int pthread_cond_signal(pthread_cond_t *);
extern int pthread_cond_broadcast(pthread_cond_t *);
extern int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
extern int pthread_cond_timedwait(pthread_cond_t *, pthread_mutex_t *,
                                  const struct timespec *);

extern int pthread_key_create(pthread_key_t *, void (*)(void *));
extern void *pthread_getspecific(pthread_key_t);
extern int pthread_setspecific(pthread_key_t, const void *);

extern int pthread_create(pthread_t *, const pthread_attr_t *,
                          void *(*)(void *), void *);
extern int pthread_join(pthread_t, void **);
extern int pthread_detach(pthread_t);
extern pthread_t pthread_self(void);

extern int pthread_once(pthread_once_t *, void (*)(void));

#define PTHREAD_MUTEX_RECURSIVE 1
#define PTHREAD_MUTEX_INITIALIZER                                              \
  {}
#define PTHREAD_COND_INITIALIZER                                               \
  {}

extern int sched_yield(void);

typedef struct {
} mbstate_t;

#ifdef _WIN32
extern bool isinf(float num);
extern bool isinf(double num);
extern bool isinf(long double num);

typedef struct {
} FILE;

extern bool __is_windows_terminal(FILE *);

extern void *_aligned_malloc(size_t, size_t);
extern void _aligned_free(void *);
#endif
