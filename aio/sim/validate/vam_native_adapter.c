/* Entry points for the native VAM plant (Chen, Dave, McAvoy, Luyben 2003, the C files of the UMD site), built
   by vam_native_record.py into a DLL beside the unmodified unit-operation sources (VAModel.c without its MEX
   gateway). Explicit Euler as in their test_VAcPlant.m. */
#include <string.h>

void VAModel(double *dstatesdt, double *states_out, double *MVs_out, double *measurements,
             double states[], double MVs[], double time, int is_initial, int disturbance_ID);
void steadystate(double MVS[], double STATES[]);

#define NX 246
#define NU 26
#define NY 43

__declspec(dllexport) void vam_init(double *x, double *u) { steadystate(u, x); }

/* derivatives and the 43 measurements at x with the MVs u; time in minutes (catalyst age) */
__declspec(dllexport) void vam_f(const double *x, const double *u, double t, int dist, double *dx, double *y)
{
    double xs[NX], us[NU], xo[NX], uo[NU];
    memcpy(xs, x, sizeof xs); memcpy(us, u, sizeof us);
    VAModel(dx, xo, uo, y, xs, us, t, 0, dist);
}

/* n Euler steps of h minutes from step k0 (time k*h); y: the measurements of the last evaluation */
__declspec(dllexport) void vam_euler(double *x, const double *u, long long k0, int n, double h, int dist, double *y)
{
    double dx[NX];
    int i, j;
    for (i = 0; i < n; i++) {
        vam_f(x, u, (double)(k0+i)*h, dist, dx, y);
        for (j = 0; j < NX; j++) x[j] += h*dx[j];
    }
}
