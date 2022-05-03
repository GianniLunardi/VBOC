/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) double_pendulum_ode_impl_dae_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};

/* double_pendulum_ode_impl_dae_fun:(i0[4],i1[4],i2[2],i3[],i4[])->(o0[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][1] : 0;
  a2=arg[0]? arg[0][3] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][2] : 0;
  a3=2.0480000000000007e-01;
  a4=casadi_sq(a1);
  a3=(a3*a4);
  a4=-2.;
  a5=arg[0]? arg[0][1] : 0;
  a6=(a4*a5);
  a7=2.;
  a8=arg[0]? arg[0][0] : 0;
  a9=(a7*a8);
  a6=(a6+a9);
  a6=sin(a6);
  a3=(a3*a6);
  a6=8.0000000000000004e-01;
  a9=arg[2]? arg[2][1] : 0;
  a10=(a7*a9);
  a11=(a8-a5);
  a11=cos(a11);
  a10=(a10*a11);
  a10=(a6*a10);
  a3=(a3+a10);
  a10=4.0000000000000002e-01;
  a11=9.8100000000000005e+00;
  a12=(a4*a5);
  a12=(a12+a8);
  a12=sin(a12);
  a12=(a11*a12);
  a12=(a6*a12);
  a12=(a10*a12);
  a12=(a12/a7);
  a13=(a8-a5);
  a13=sin(a13);
  a14=casadi_sq(a2);
  a13=(a13*a14);
  a13=(a6*a13);
  a13=(a6*a13);
  a13=(a10*a13);
  a12=(a12+a13);
  a13=4.7088000000000010e+00;
  a14=sin(a8);
  a13=(a13*a14);
  a12=(a12+a13);
  a13=arg[2]? arg[2][0] : 0;
  a12=(a12-a13);
  a12=(a7*a12);
  a12=(a6*a12);
  a3=(a3+a12);
  a12=6.4000000000000012e-01;
  a3=(a3/a12);
  a3=(a3/a6);
  a14=(a4*a5);
  a15=(a7*a8);
  a14=(a14+a15);
  a14=cos(a14);
  a14=(a10*a14);
  a14=(a14-a6);
  a14=(a14-a10);
  a3=(a3/a14);
  a0=(a0-a3);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][3] : 0;
  a3=-2.0090880000000007e+00;
  a14=(a7*a8);
  a14=(a14-a5);
  a14=sin(a14);
  a3=(a3*a14);
  a14=8.1920000000000034e-02;
  a2=casadi_sq(a2);
  a14=(a14*a2);
  a2=(a4*a5);
  a15=(a7*a8);
  a2=(a2+a15);
  a2=sin(a2);
  a14=(a14*a2);
  a3=(a3-a14);
  a1=casadi_sq(a1);
  a1=(a7*a1);
  a1=(a12*a1);
  a1=(a6*a1);
  a1=(a10*a1);
  a1=(a6*a1);
  a14=(a8-a5);
  a14=sin(a14);
  a1=(a1*a14);
  a3=(a3-a1);
  a13=(a7*a13);
  a1=(a8-a5);
  a1=cos(a1);
  a13=(a13*a1);
  a13=(a6*a13);
  a13=(a10*a13);
  a3=(a3+a13);
  a13=sin(a5);
  a11=(a11*a13);
  a11=(a6*a11);
  a11=(a10*a11);
  a9=(a7*a9);
  a11=(a11-a9);
  a11=(a12*a11);
  a3=(a3+a11);
  a3=(a3/a12);
  a3=(a3/a6);
  a3=(a3/a10);
  a4=(a4*a5);
  a7=(a7*a8);
  a4=(a4+a7);
  a4=cos(a4);
  a4=(a10*a4);
  a4=(a4-a6);
  a4=(a4-a10);
  a3=(a3/a4);
  a0=(a0-a3);
  if (res[0]!=0) res[0][3]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void double_pendulum_ode_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_ode_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int double_pendulum_ode_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real double_pendulum_ode_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_ode_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* double_pendulum_ode_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_ode_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* double_pendulum_ode_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int double_pendulum_ode_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
