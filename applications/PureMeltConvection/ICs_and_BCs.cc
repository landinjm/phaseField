// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

template <int dim, int degree>
void customPDE<dim, degree>::setInitialCondition(const dealii::Point<dim>& p, const unsigned int index, double& scalar_IC, dealii::Vector<double>& vector_IC)
{
    // ---------------------------------------------------------------------
    // ENTER THE INITIAL CONDITIONS HERE
    // ---------------------------------------------------------------------
    // Enter the function describing conditions for the fields at point "p".
    // Use "if" statements to set the initial condition for each variable
    // according to its variable index

    // The initial condition is a set of overlapping circles/spheres defined
    // by a hyperbolic tangent function. The center of each circle/sphere is
    // given by "center" and its radius is given by "radius".
    double center[3] = { 0.5, 0.0, 0.0 };
    scalar_IC = 0;
    double dist = 0.0;
    double rad = 10.0;
    for (unsigned int dir = 0; dir < dim; dir++) {
        dist += (p[dir] - center[dir] * userInputs.domain_size[dir]) * (p[dir] - center[dir] * userInputs.domain_size[dir]);
    }
    dist = std::sqrt(dist);
    double particle = 0.5 * (1.0 - std::tanh((dist - rad) / (W * std::sqrt(2))));

    // Velocity
    if (index == 0) {
        for (unsigned int d = 0; d < dim; d++) {
            vector_IC(d) = 0.0;
        }
    }
    // Pressure
    else if (index == 1) {
        scalar_IC = 0.0;
    }
    // Solid-liquid Order
    else if (index == 2) {
        scalar_IC = 2.0 * particle - 1.0;
    }
    // Dimensionless Temperature
    else if (index == 4) {
        scalar_IC = -0.55;
    }

    // ---------------------------------------------------------------------
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void customPDE<dim, degree>::setNonUniformDirichletBCs(const dealii::Point<dim>& p, const unsigned int index, const unsigned int direction, const double time, double& scalar_BC, dealii::Vector<double>& vector_BC)
{
    // --------------------------------------------------------------------------
    // ENTER THE NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS HERE
    // --------------------------------------------------------------------------
    // Enter the function describing conditions for the fields at point "p".
    // Use "if" statements to set the boundary condition for each variable
    // according to its variable index. This function can be left blank if there
    // are no non-uniform Dirichlet boundary conditions. For BCs that change in
    // time, you can access the current time through the variable "time". The
    // boundary index can be accessed via the variable "direction", which starts
    // at zero and uses the same order as the BC specification in parameters.in
    // (i.e. left = 0, right = 1, bottom = 2, top = 3, front = 4, back = 5).

    /*if(index == 0){
        if(direction == 0){
            double normalizedPos = p[1]/userInputs.domain_size[1];
            vector_BC(0) = 4.0*normalizedPos*(1.0-normalizedPos);
        }
    }*/

    // -------------------------------------------------------------------------
}
