// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
// This function sets attributes for each variable/equation in the app. The
// attributes are set via standardized function calls. The first parameter for each
// function call is the variable index (starting at zero). The first set of
// variable/equation attributes are the variable name (any string), the variable
// type (SCALAR/VECTOR), and the equation type (EXPLICIT_TIME_DEPENDENT/
// TIME_INDEPENDENT/AUXILIARY). The next set of attributes describe the
// dependencies for the governing equation on the values and derivatives of the
// other variables for the value term and gradient term of the RHS and the LHS.
// The final pair of attributes determine whether a variable represents a field
// that can nucleate and whether the value of the field is needed for nucleation
// rate calculations.

void variableAttributeLoader::loadVariableAttributes()
{
    // Variable 0
    set_variable_name(0, "P");
    set_variable_type(0, SCALAR);
    set_variable_equation_type(0, TIME_INDEPENDENT);

    set_dependencies_value_term_RHS(0, "grad(P)");
    set_dependencies_gradient_term_RHS(0, "grad(P)");
    set_dependencies_value_term_LHS(0, "");
    set_dependencies_gradient_term_LHS(0, "grad(change(P))");
    // Variable 1
    set_variable_name(1, "dummy");
    set_variable_type(1, SCALAR);
    set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

    set_dependencies_value_term_RHS(1, "dummy");
    set_dependencies_gradient_term_RHS(1, "");
}

// =============================================================================================
// explicitEquationRHS (needed only if one or more equation is explict time dependent)
// =============================================================================================
// This function calculates the right-hand-side of the explicit time-dependent
// equations for each variable. It takes "variable_list" as an input, which is a list
// of the value and derivatives of each of the variables at a specific quadrature
// point. The (x,y,z) location of that quadrature point is given by "q_point_loc".
// The function outputs two terms to variable_list -- one proportional to the test
// function and one proportional to the gradient of the test function. The index for
// each variable in this list corresponds to the index given at the top of this file.

template <int dim, int degree>
void customPDE<dim, degree>::explicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const
{

    scalarvalueType dummy = variable_list.get_scalar_value(1);
    variable_list.set_scalar_value_term_RHS(1, dummy);
}

// =============================================================================================
// nonExplicitEquationRHS (needed only if one or more equation is time independent or auxiliary)
// =============================================================================================
// This function calculates the right-hand-side of all of the equations that are not
// explicit time-dependent equations. It takes "variable_list" as an input, which is
// a list of the value and derivatives of each of the variables at a specific
// quadrature point. The (x,y,z) location of that quadrature point is given by
// "q_point_loc". The function outputs two terms to variable_list -- one proportional
// to the test function and one proportional to the gradient of the test function. The
// index for each variable in this list corresponds to the index given at the top of
// this file.

template <int dim, int degree>
void customPDE<dim, degree>::nonExplicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const
{

    // Grab derivative model variable
    scalargradType Px = variable_list.get_scalar_gradient(0);

    // Variable coeffcient a(x)
    scalarvalueType ax = constV(1.0) / (constV(0.05) + constV(2.0) * q_point_loc.square());

    // Initialize submission terms
    scalarvalueType eq_P = constV(1.0);
    scalargradType eqx_P = -ax * Px;

    // Submitting the terms for the governing equations
    variable_list.set_scalar_value_term_RHS(0, eq_P);
    variable_list.set_scalar_gradient_term_RHS(0, eqx_P);
}

// =============================================================================================
// equationLHS (needed only if at least one equation is time independent)
// =============================================================================================
// This function calculates the left-hand-side of time-independent equations. It
// takes "variable_list" as an input, which is a list of the value and derivatives of
// each of the variables at a specific quadrature point. The (x,y,z) location of that
// quadrature point is given by "q_point_loc". The function outputs two terms to
// variable_list -- one proportional to the test function and one proportional to the
// gradient of the test function -- for the left-hand-side of the equation. The index
// for each variable in this list corresponds to the index given at the top of this
// file. If there are multiple elliptic equations, conditional statements should be
// sed to ensure that the correct residual is being submitted. The index of the field
// being solved can be accessed by "this->currentFieldIndex".

template <int dim, int degree>
void customPDE<dim, degree>::equationLHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const
{

    // --- Getting the values and derivatives of the model variables ---
    scalargradType DPx = variable_list.get_change_in_scalar_gradient(0);

    // Variable coeffcient a(x)
    scalarvalueType ax = constV(1.0) / (constV(0.05) + constV(2.0) * q_point_loc.square());

    // Initialize submission terms
    scalargradType eqx_DP = ax * DPx;

    // --- Submitting the terms for the governing equations ---
    variable_list.set_scalar_gradient_term_LHS(0, eqx_DP);
}
