// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
// This function sets attributes for each variable/equation in the app. The
// attributes are set via standardized function calls. The first parameter for
// each function call is the variable index (starting at zero). The first set of
// variable/equation attributes are the variable name (any string), the variable
// type (Scalar/Vector), and the equation type (ExplicitTimeDependent/
// TimeIndependent/Auxiliary). The next set of attributes describe the
// dependencies for the governing equation on the values and derivatives of the
// other variables for the value term and gradient term of the RHS and the LHS.
// The final pair of attributes determine whether a variable represents a field
// that can nucleate and whether the value of the field is needed for nucleation
// rate calculations.

void
CustomAttributeLoader::load_variable_attributes()
{
  // Variable 0
  set_variable_name(0, "c");
  set_variable_type(0, Scalar);
  set_variable_equation_type(0, ExplicitTimeDependent);

  set_dependencies_value_term_rhs(0, "c");
  set_dependencies_gradient_term_rhs(0, "n,grad(c)");

  // Variable 1
  set_variable_name(1, "n");
  set_variable_type(1, Scalar);
  set_variable_equation_type(1, ExplicitTimeDependent);

  set_dependencies_value_term_rhs(1, "c,n");
  set_dependencies_gradient_term_rhs(1, "grad(n)");
}

// =============================================================================================
// explicitEquationRHS (needed only if one or more equation is explict time
// dependent)
// =============================================================================================
// This function calculates the right-hand-side of the explicit time-dependent
// equations for each variable. It takes "variable_list" as an input, which is a
// list of the value and derivatives of each of the variables at a specific
// quadrature point. The (x,y,z) location of that quadrature point is given by
// "q_point_loc". The function outputs two terms to variable_list -- one
// proportional to the test function and one proportional to the gradient of the
// test function. The index for each variable in this list corresponds to the
// index given at the top of this file.

template <int dim, int degree>
void
CustomPDE<dim, degree>::explicitEquationRHS(
  [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{
  // --- Getting the values and derivatives of the model variables ---

  // c
  scalarvalueType c  = variable_list.template get_value<ScalarValue>(0);
  scalargradType  cx = variable_list.template get_gradient<ScalarGrad>(0);

  // n
  scalarvalueType n  = variable_list.template get_value<ScalarValue>(1);
  scalargradType  nx = variable_list.template get_gradient<ScalarGrad>(1);

  // --- Setting the expressions for the terms in the governing equations ---

  // Free energy for each phase and their first and second derivatives
  scalarvalueType fa   = constV(2.0) * c * c;
  scalarvalueType fac  = constV(4.0) * c;
  scalarvalueType facc = constV(4.0);
  scalarvalueType fb   = constV(2.0) * (c * c - 2.0 * c + constV(1.0));
  scalarvalueType fbc  = constV(4.0) * (c - 1.0);
  scalarvalueType fbcc = constV(4.0);

  // Interpolation function and its derivative
  scalarvalueType h  = (3.0 * n * n - 2.0 * n * n * n);
  scalarvalueType hn = (6.0 * n - 6.0 * n * n);

  // Residual equations
  scalargradType  mux   = (cx * ((1.0 - h) * facc + h * fbcc) + nx * ((fbc - fac) * hn));
  scalarvalueType eq_c  = c;
  scalargradType  eqx_c = (constV(-Mc * userInputs.dtValue) * mux);
  scalarvalueType eq_n  = (n - constV(userInputs.dtValue * Mn) * (fb - fa) * hn);
  scalargradType  eqx_n = (constV(-userInputs.dtValue * Kn * Mn) * nx);

  // --- Submitting the terms for the governing equations ---

  // Terms for the equation to evolve the concentration
  variable_list.set_scalar_value_term_rhs(0, eq_c);
  variable_list.set_scalar_gradient_term_rhs(0, eqx_c);

  // Terms for the equation to evolve the order parameter
  variable_list.set_scalar_value_term_rhs(1, eq_n);
  variable_list.set_scalar_gradient_term_rhs(1, eqx_n);
}

// =============================================================================================
// nonExplicitEquationRHS (needed only if one or more equation is time
// independent or auxiliary)
// =============================================================================================
// This function calculates the right-hand-side of all of the equations that are
// not explicit time-dependent equations. It takes "variable_list" as an input,
// which is a list of the value and derivatives of each of the variables at a
// specific quadrature point. The (x,y,z) location of that quadrature point is
// given by "q_point_loc". The function outputs two terms to variable_list --
// one proportional to the test function and one proportional to the gradient of
// the test function. The index for each variable in this list corresponds to
// the index given at the top of this file.

template <int dim, int degree>
void
CustomPDE<dim, degree>::nonExplicitEquationRHS(
  [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{}

// =============================================================================================
// equationLHS (needed only if at least one equation is time independent)
// =============================================================================================
// This function calculates the left-hand-side of time-independent equations. It
// takes "variable_list" as an input, which is a list of the value and
// derivatives of each of the variables at a specific quadrature point. The
// (x,y,z) location of that quadrature point is given by "q_point_loc". The
// function outputs two terms to variable_list -- one proportional to the test
// function and one proportional to the gradient of the test function -- for the
// left-hand-side of the equation. The index for each variable in this list
// corresponds to the index given at the top of this file. If there are multiple
// elliptic equations, conditional statements should be sed to ensure that the
// correct residual is being submitted. The index of the field being solved can
// be accessed by "this->currentFieldIndex".

template <int dim, int degree>
void
CustomPDE<dim, degree>::equationLHS(
  [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{}