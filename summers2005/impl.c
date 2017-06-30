/* Copyright 2017 Peter Williams and collaborators
 * Licensed under the MIT License.
 */

#include <Python.h>

#include <stdio.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_poly.h>


#define COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

typedef enum result_t {
    RESULT_OK = 0,
    RESULT_LAMBDA_FAILED = 1,
    RESULT_X_FAILED = 2,
    RESULT_GSL_ERROR = 3,
} result_t;


typedef enum handedness_t {
    R_MODE = 1,
    L_MODE = -1,
} handedness_t;


typedef enum integration_type_t {
    ITYPE_BOUNCE_AVERAGED,
    ITYPE_LOCAL,
} integration_type_t;


typedef struct parameters_t {
    double E; /* dimensionless kinetic energy */
    double sin_alpha; /* sine of the pitch angle */
    double Omega_e; /* local electron gyrofrequency in rad/s */
    handedness_t handedness; /* handedness of wave; `s` in Summers/Shprits notation */
    double alpha_star; /* (cyclotron freq / plasma freq)**2 */
    double R; /* magnetic wave energy perturbation: (delta B / B)**2 */
    double x_m; /* center of wave frequency spectrum in units of cyclotron freq */
    double delta_x; /* width of wave frequency spectrum in units of cyclotron freq */
} parameters_t;


typedef struct state_t {
    parameters_t p;
    integration_type_t itype;
    result_t last_error;

    double gamma; /* Lorentz factor */
    double a; /* See Summers 2005, just after Eqn 24 */
    double b; /* See Summers 2005, just after Eqn 25 */
    double beta; /* velocity over speed of light */

    double latitude; /* current magnetic latitude, radians */
    double sin_alpha; /* current sin(alpha); varies since mu is conserved */
    double mu; /* current cos(alpha) */
    double Omega_e; /* current Omega_e; varies with latitude since it scales as B */

    int n_xy; /* Number of resonances encountered */
    double x[3]; /* x values of resonances */
    double y[3]; /* y values of resonances */
    double Q[3]; /* (1 - x cos alpha / y beta) */
    double R[3]; /* complex expression in summand */
} state_t;


typedef struct coefficients_t {
    double dimensionless_p; /* momentum over (m_0 * c) */
    double Daa;
    double err_Daa;
    double Dap_on_p;
    double err_Dpp_on_p2;
    double Dpp_on_p2;
    double err_Dap_on_p;
} coefficients_t;


static state_t *global_context = NULL;
static char global_err_msg[1024] = "";

static void
s05_error_handler(const char *reason, const char *file, int line, int gsl_errno)
{
    /* Go ahead and segfault if global_context is NULL. */

    int n;

    n = snprintf(global_err_msg, COUNT(global_err_msg),
                 "%s (%s:%d; %d = %s)", reason, file, line, gsl_errno, gsl_strerror(gsl_errno));
    if ((size_t) n > COUNT(global_err_msg))
        n = COUNT(global_err_msg) - 1;
    global_err_msg[n] = '\0';

    global_context->last_error = RESULT_GSL_ERROR;
}

void
summers2005_debug_hook(void)
{
    printf("Debug hook called.\n");
}

static gsl_integration_workspace *integ_workspace = NULL; /* We're lame and leak these */
static gsl_poly_complex_workspace *poly5_workspace = NULL;
static gsl_poly_complex_workspace *poly7_workspace = NULL;

const size_t INTEG_WS_SIZE = 512;


const int lambda = -1; /* signifies that we're looking at electrons */
const double epsilon = 0.0005446; /* electron-to-proton mass ratio */
const double epsm1 =  -0.9994554; /* epsilon - 1; C is lame and can't do the math with consts! */
const double sigma = 2.; /* range of `x` that we consider */
const double pi_on_2nu = 0.89039; /* pi/2nu, nu = sqrt(pi)erf(sigma); see after Shprits 06 Eqn 5 */


static inline result_t
apply_latitude(double latitude, state_t *state)
{
    int i;

    if (state->last_error != RESULT_OK)
        return state->last_error;

    state->latitude = latitude;

    double scl = cos(latitude); /* sin of colatitude = cos of latitude */
    double r = sqrt(4 - 3 * scl * scl) / pow(scl, 6); /* B(lam) / B_eq */

    state->Omega_e = state->p.Omega_e * r;
    state->sin_alpha = state->p.sin_alpha * sqrt(r);
    state->mu = sqrt(1 - state->sin_alpha * state->sin_alpha); /* cos(alpha) */

    if (state->mu == 0.) /* happens at the very top of our bounce trajectory */
        state->mu = 1e-20;

    /* Find the critical `x` and `y` values. We do this following Appendix A
     * and Equation 24 of Summers 2005. */

    state->n_xy = 0;

    double c[5] = { 0 };
    double z[8] = { 0 };

    const int s = state->p.handedness;
    const double a = state->a;
    const double a2 = a * a;
    const double bm2 = pow(state->beta * state->mu, 2);

    c[0] = -a2 * epsilon;
    c[1] = a2 * s * epsm1 - 2 * a * epsilon;
    c[2] = a2 + 2 * a * s * epsm1 - epsilon + bm2 * (state->b + epsilon);
    c[3] = 2 * a + s * epsm1 - bm2 * s * epsm1;
    c[4] = 1 - bm2;

    gsl_poly_complex_solve(c, 5, poly5_workspace, z);

    if (state->last_error != RESULT_OK) {
        strncat(global_err_msg, " (while finding x/y roots)", COUNT(global_err_msg));
        return state->last_error;
    }

    for (i = 0; i < 4; i++) {
        if (z[2*i + 1] != 0.) /* Imaginary root? */
            continue;

        double x = z[2*i];

        if (x < 0) /* Non-physical root? */
            continue;

        if (x < state->p.x_m - sigma * state->p.delta_x || x > state->p.x_m + sigma * state->p.delta_x)
            continue; /* Out of waveband? */

        double y = (x + state->a) / (state->beta * state->mu);

        if (y > 0) /* TEMP ~"wave propagating in forward direction"? */
            continue;

        state->x[state->n_xy] = x;
        state->y[state->n_xy] = y;
        state->n_xy++;
    }

    if (state->n_xy > 3) {
        snprintf(global_err_msg, COUNT(global_err_msg), "expect 0 to 3 X/Y solutions; got %d", state->n_xy);
        return RESULT_X_FAILED;
    }

    /* Now we calculate F(x,y) = dx/dy, following Appendix C of Summers 2005,
     * and related values that contribute to the sums in the integrands. `Q`
     * is `1 - x cosa / y beta` and `R` is the factor that is in common
     * between all three calculations. */

    for (i = 0; i < state->n_xy; i++) {
        double x = state->x[i];
        double y = state->y[i];

        c[0] = epsilon * (state->b + epsilon);
        c[1] = -0.5 * s * epsm1 * (state->b + 4 * epsilon);
        c[2] = 1 - 4 * epsilon + epsilon * epsilon;
        c[3] = 2 * s * epsm1;
        c[4] = 1.;

        double g = gsl_poly_eval(c, 5, x);
        double F = y * pow((x - s) * (x + s * epsilon), 2) / (x * g);

        state->Q[i] = 1 - x * state->mu / (y * state->beta);

        double k = pow((x - state->p.x_m) / state->p.delta_x, 2);
        state->R[i] = state->p.R * fabs(F) * exp(-k) / (state->p.delta_x * fabs(state->beta * state->mu - F));

        if (!isfinite(state->R[i]))
            summers2005_debug_hook();
    }

    return RESULT_OK;
}


typedef double (*gsl_signature)(double, void *);

static double
daa_integrand(double latitude, state_t *state)
{
    result_t r;

    if (state->last_error != RESULT_OK)
        return 0;

    if ((r = apply_latitude(latitude, state)) != RESULT_OK) {
        state->last_error = r;
        return 0;
    }

    int i;
    double v = 0;

    for (i = 0; i < state->n_xy; i++)
        v += state->R[i] * pow(state->Q[i], 2);

    if (state->itype == ITYPE_BOUNCE_AVERAGED)
        v *= state->mu * pow(cos(state->latitude), 7);

    return v * state->Omega_e;
}


static double
dap_on_p_integrand(double latitude, state_t *state)
{
    result_t r;

    if (state->last_error != RESULT_OK)
        return 0;

    if ((r = apply_latitude(latitude, state)) != RESULT_OK) {
        state->last_error = r;
        return 0;
    }

    int i;
    double v = 0;

    for (i = 0; i < state->n_xy; i++)
        v += state->R[i] * state->Q[i] * state->x[i] / state->y[i];

    /* Here we transform the sqrt(1 + 3 sin^2 lambda) term to use the cos
     * instead */

    double cos_lam = cos(state->latitude);
    double sa = state->sin_alpha;

    if (state->itype == ITYPE_BOUNCE_AVERAGED)
        v *= state->mu * cos_lam * sqrt(4 - 3 * cos_lam * cos_lam) / sa;

    return v * state->Omega_e * sa;
}


static double
dpp_on_p2_integrand(double latitude, state_t *state)
{
    result_t r;

    if (state->last_error != RESULT_OK)
        return 0;

    if ((r = apply_latitude(latitude, state)) != RESULT_OK) {
        state->last_error = r;
        return 0;
    }

    int i;
    double v = 0;

    for (i = 0; i < state->n_xy; i++)
        v += state->R[i] * pow(state->x[i] / state->y[i], 2);

    /* Same transform as in dap_on_p. */

    double cos_lam = cos(state->latitude);
    double sa = state->sin_alpha;

    if (state->itype == ITYPE_BOUNCE_AVERAGED)
        v *= cos_lam * sqrt(4 - 3 * cos_lam * cos_lam) / state->mu;

    return v * state->Omega_e * sa * sa;
}


static result_t
calc_coefficients(parameters_t *params, coefficients_t *coeffs)
{
    gsl_error_handler_t *prev_handler;
    gsl_function integrand;
    state_t state;
    double c[7] = { 0 };
    double z[12] = { 0 };
    double lambda_m = -1;
    double result, abserr, k;
    int i;

    if (params->sin_alpha == 1.0) {
        /* Integration blows up in this case. */
        state.gamma = params->E + 1;
        state.beta = sqrt(params->E * (params->E + 2)) / (params->E + 1);
        coeffs->dimensionless_p = state.gamma * state.beta;

        coeffs->Daa = 0;
        coeffs->err_Daa = 0;
        coeffs->Dap_on_p = -0.;
        coeffs->err_Dap_on_p = 0;
        coeffs->Dpp_on_p2 = 0;
        coeffs->err_Dpp_on_p2 = 0;
        return RESULT_OK;
    }

    global_context = &state;
    prev_handler = gsl_set_error_handler(s05_error_handler);

    if (integ_workspace == NULL) {
        integ_workspace = gsl_integration_workspace_alloc(INTEG_WS_SIZE);
        poly5_workspace = gsl_poly_complex_workspace_alloc(5);
        poly7_workspace = gsl_poly_complex_workspace_alloc(7);
    }

    state.p = *params;
    state.itype = ITYPE_BOUNCE_AVERAGED;
    state.last_error = RESULT_OK;

    /* Figure out the mirroring latitude; Shprits (2006) equation 10. */

    double f0 = pow(state.p.sin_alpha, 4);
    c[0] = -4 * f0;
    c[1] = 3 * f0;
    c[6] = 1.;

    gsl_poly_complex_solve(c, 7, poly7_workspace, z);

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while finding lambda_m)", COUNT(global_err_msg));
        return state.last_error;
    }

    for (i = 0; i < 6; i++) {
        if (z[2*i + 1] == 0. && z[2*i] > 0) {
            lambda_m = acos(sqrt(z[2*i]));
            break;
        }
    }

    if (!(lambda_m >= 0 && lambda_m <= M_PI_2)) {
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        snprintf(global_err_msg, COUNT(global_err_msg), "failed to compute lambda_m");
        return RESULT_LAMBDA_FAILED;
    }

    /* It can happen that lambda_m is just a tiiiiiny bit too large such that
     * when we compute the sine of the pitch angle at lambda_m, we get
     * something bigger than one. Witness our beautiful workaround.
     */

    double cos_lat = cos(lambda_m);

    if (sqrt(4 - 3 * cos_lat * cos_lat) / (pow(cos_lat, 6) * state.p.sin_alpha * state.p.sin_alpha) > 1.)
        lambda_m -= 1e-7;

    /* TEMP lat limit of 15 degrees*/

    if (lambda_m > 0.2618)
        lambda_m = 0.2618;

    /* Pre-compute quantities that do not depend on lambda. When lambda
     * changes, B and alpha change. Consequently Omega_e changes too. In our
     * model the other parameters stay fixed.
     */

    state.gamma = state.p.E + 1;
    state.a = ((int) state.p.handedness) * lambda / state.gamma;
    state.b = (1 + epsilon) / state.p.alpha_star;
    state.beta = sqrt(state.p.E * (state.p.E + 2)) / (state.p.E + 1);
    coeffs->dimensionless_p = state.gamma * state.beta;

    /* Shprits 2006 eqn 9, credited to Lencheck+ 1971 and Shultz & Lanzerotti 1974: */

    double s0 = 1.38 - 0.32 * (params->sin_alpha + sqrt(params->sin_alpha));
    double f1 = pi_on_2nu / (pow(state.p.E + 1, 2) * s0);

    /* Let's integrate! First, D_aa. */

    integrand.function = (gsl_signature) daa_integrand;
    integrand.params = &state;

    gsl_integration_qag(
        &integrand,
        0., /* lower limit */
        lambda_m, /* upper limit */
        0., /* absolute error tolerance; zero means "don't apply" */
        1e-6, /* relative error tolerance */
        INTEG_WS_SIZE, /* allocated workspace size */
        GSL_INTEG_GAUSS51, /* integration rule */
        integ_workspace, /* workspace */
        &result, /* output: value of the integral */
        &abserr /* estimated absolute error */
    );

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Daa)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    k = f1 / (1 - state.p.sin_alpha * state.p.sin_alpha);
    coeffs->Daa = result * k;
    coeffs->err_Daa = abserr * k;

    /* D_ap/p. */

    integrand.function = (gsl_signature) dap_on_p_integrand;

    gsl_integration_qag(
        &integrand,
        0., /* lower limit */
        lambda_m, /* upper limit */
        0., /* absolute error tolerance; zero means "don't apply" */
        1e-6, /* relative error tolerance */
        INTEG_WS_SIZE, /* allocated workspace size */
        GSL_INTEG_GAUSS51, /* integration rule */
        integ_workspace, /* workspace */
        &result, /* output: value of the integral */
        &abserr /* estimated absolute error */
    );

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Dap)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    k = -f1 * state.p.sin_alpha / (state.beta * sqrt(1 - state.p.sin_alpha * state.p.sin_alpha));
    coeffs->Dap_on_p = result * k;
    coeffs->err_Dap_on_p = abserr * fabs(k);

    /* D_pp/p^2. */

    integrand.function = (gsl_signature) dpp_on_p2_integrand;

    gsl_integration_qag(
        &integrand,
        0., /* lower limit */
        lambda_m, /* upper limit */
        0., /* absolute error tolerance; zero means "don't apply" */
        1e-6, /* relative error tolerance */
        INTEG_WS_SIZE, /* allocated workspace size */
        GSL_INTEG_GAUSS51, /* integration rule */
        integ_workspace, /* workspace */
        &result, /* output: value of the integral */
        &abserr /* estimated absolute error */
    );

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Dpp)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    k = f1 * pow(state.beta, -2);
    coeffs->Dpp_on_p2 = result * k;
    coeffs->err_Dpp_on_p2 = abserr * k;

    /* Huzzah, all done. */

    gsl_set_error_handler(prev_handler);
    global_context = NULL;
    return RESULT_OK;
}


static result_t
calc_unaveraged_coefficients(parameters_t *params, coefficients_t *coeffs)
{
    gsl_error_handler_t *prev_handler;
    state_t state;

    global_context = &state;
    prev_handler = gsl_set_error_handler(s05_error_handler);

    if (integ_workspace == NULL) {
        integ_workspace = gsl_integration_workspace_alloc(INTEG_WS_SIZE);
        poly5_workspace = gsl_poly_complex_workspace_alloc(5);
        poly7_workspace = gsl_poly_complex_workspace_alloc(7);
    }

    state.p = *params;
    state.itype = ITYPE_LOCAL;
    state.last_error = RESULT_OK;

    /* The structure here is paralleling calc_coefficients for maintainability. */

    state.gamma = state.p.E + 1;
    state.a = ((int) state.p.handedness) * lambda / state.gamma;
    state.b = (1 + epsilon) / state.p.alpha_star;
    state.beta = sqrt(state.p.E * (state.p.E + 2)) / (state.p.E + 1);
    coeffs->dimensionless_p = state.gamma * state.beta;

    double f1 = pi_on_2nu / pow(state.p.E + 1, 2);

    /* Compute! It all works if we pretend latitude = 0 and set `itype` to LOCAL. */

    coeffs->Daa = daa_integrand(0., &state);

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Daa)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    coeffs->Daa *= f1;
    coeffs->err_Daa = 0;

    /* D_ap/p. */

    coeffs->Dap_on_p = dap_on_p_integrand(0., &state);

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Dap)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    coeffs->Dap_on_p *= f1 / state.beta;
    coeffs->err_Dap_on_p = 0;

    /* D_pp/p^2. */

    coeffs->Dpp_on_p2 = dpp_on_p2_integrand(0., &state);

    if (state.last_error != RESULT_OK) {
        strncat(global_err_msg, " (while computing Dpp)", COUNT(global_err_msg));
        gsl_set_error_handler(prev_handler);
        global_context = NULL;
        return state.last_error;
    }

    coeffs->Dpp_on_p2 *= f1 * pow(state.beta, -2);
    coeffs->err_Dpp_on_p2 = 0.;

    /* Huzzah, all done. */

    gsl_set_error_handler(prev_handler);
    global_context = NULL;
    return RESULT_OK;
}


static PyObject*
get_coeffs(PyObject *self, PyObject* args)
{
    int modespec;
    int handspec;
    parameters_t params;
    coefficients_t coeffs = { 0 };
    result_t r;

    if (!PyArg_ParseTuple(args, "iiddddddd", &modespec, &handspec,
                          &params.E,
                          &params.sin_alpha,
                          &params.Omega_e,
                          &params.alpha_star,
                          &params.R,
                          &params.x_m,
                          &params.delta_x))
        return NULL;

    if (handspec == 0)
        params.handedness = R_MODE;
    else if (handspec == 1)
        params.handedness = L_MODE;
    else {
        PyErr_SetString(PyExc_RuntimeError, "unexpected handedness magic constant");
        return NULL;
    }

    if (modespec == 0)
        r = calc_coefficients(&params, &coeffs);
    else if (modespec == 1)
        r = calc_unaveraged_coefficients(&params, &coeffs);
    else {
        PyErr_SetString(PyExc_RuntimeError, "unexpected mode magic constant");
        return NULL;
    }

    if (r != RESULT_OK) {
        PyErr_SetString(PyExc_RuntimeError, global_err_msg);
        return NULL;
    }

    return Py_BuildValue("ddddddd", coeffs.dimensionless_p,
                         coeffs.Daa, coeffs.err_Daa,
                         coeffs.Dap_on_p, coeffs.err_Dap_on_p,
                         coeffs.Dpp_on_p2, coeffs.err_Dpp_on_p2);
}


static PyMethodDef methods[] = {
    { "get_coeffs", get_coeffs, METH_VARARGS|METH_KEYWORDS,
      "(mode, handedness, E, sa, Oe, a*, R, xm dx) -> (dp, daa, udaa, dap/p, udap/p, dpp/p2, udpp/p2)" },
    { NULL, NULL, METH_NOARGS, NULL },
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_impl",
    NULL,
    0,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

# define INIT_RET_TYPE PyObject *
#else
# define INIT_RET_TYPE void
#endif

INIT_RET_TYPE
PyInit__impl(void)
{
#if PY_MAJOR_VERSION >= 3
    return PyModule_Create(&module_def);
#else
    PyImport_AddModule("_impl");
    Py_InitModule("_impl", methods);
#endif
}
