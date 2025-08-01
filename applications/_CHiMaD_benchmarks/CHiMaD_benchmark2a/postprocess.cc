// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

// =============================================================================================
// loadPostProcessorVariableAttributes: Set the attributes of the postprocessing
// variables
// =============================================================================================
// This function is analogous to 'load_variable_attributes' in 'equations.h', but
// for the postprocessing expressions. It sets the attributes for each
// postprocessing expression, including its name, whether it is a vector or
// scalar (only scalars are supported at present), its dependencies on other
// variables and their derivatives, and whether to calculate an integral of the
// postprocessed quantity over the entire domain. Note: this function is not a
// member of CustomPDE.

void
CustomAttributeLoader::loadPostProcessorVariableAttributes()
{
  // Variable 0
  set_variable_name(0, "f_tot");
  set_variable_type(0, Scalar);

  set_dependencies_value_term_rhs(
    0,
    "c, grad(c), n1, n2, n3, n4, grad(n1), grad(n2), grad(n3), grad(n4)");
  set_dependencies_gradient_term_rhs(0, "");

  set_output_integral(0, true);
}

// =============================================================================================
// postProcessedFields: Set the postprocessing expressions
// =============================================================================================
// This function is analogous to 'explicitEquationRHS' and
// 'nonExplicitEquationRHS' in equations.h. It takes in "variable_list" and
// "q_point_loc" as inputs and outputs two terms in the expression for the
// postprocessing variable -- one proportional to the test function and one
// proportional to the gradient of the test function. The index for each
// variable in this list corresponds to the index given at the top of this file
// (for submitting the terms) and the index in 'equations.h' for assigning the
// values/derivatives of the primary variables.

template <int dim, int degree>
void
CustomPDE<dim, degree>::postProcessedFields(
  [[maybe_unused]] const VariableContainer<dim, degree, VectorizedArray<double>>
    &variable_list,
  [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>>
                                                            &pp_variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
  [[maybe_unused]] const VectorizedArray<double>             element_volume) const
{
  // --- Getting the values and derivatives of the model variables ---

  // The concentration and its derivatives
  scalarvalueType c  = variable_list.template get_value<ScalarValue>(0);
  scalargradType  cx = variable_list.template get_gradient<ScalarGrad>(0);

  // The order parameter and its derivatives
  scalarvalueType n1  = variable_list.template get_value<ScalarValue>(2);
  scalargradType  n1x = variable_list.template get_gradient<ScalarGrad>(2);
  scalarvalueType n2  = variable_list.template get_value<ScalarValue>(3);
  scalargradType  n2x = variable_list.template get_gradient<ScalarGrad>(3);
  scalarvalueType n3  = variable_list.template get_value<ScalarValue>(4);
  scalargradType  n3x = variable_list.template get_gradient<ScalarGrad>(4);
  scalarvalueType n4  = variable_list.template get_value<ScalarValue>(5);
  scalargradType  n4x = variable_list.template get_gradient<ScalarGrad>(5);

  // --- Setting the expressions for the terms in the postprocessing expressions
  // ---

  // Free energy for each phase and their first and second derivatives
  scalarvalueType faV = (constV(2.0) * (c - constV(0.3)) * (c - constV(0.3)));
  scalarvalueType fbV = (constV(2.0) * (c - constV(0.7)) * (c - constV(0.7)));

  // Interpolation function and its derivatives
  scalarvalueType hV =
    (n1 * n1 * n1 * (constV(6.0) * n1 * n1 - constV(15.0) * n1 + constV(10.0)) +
     n2 * n2 * n2 * (constV(6.0) * n2 * n2 - constV(15.0) * n2 + constV(10.0)) +
     n3 * n3 * n3 * (constV(6.0) * n3 * n3 - constV(15.0) * n3 + constV(10.0)) +
     n4 * n4 * n4 * (constV(6.0) * n4 * n4 - constV(15.0) * n4 + constV(10.0)));

  // Combined double-well and interaction functions (function g) and its
  // derivatives Double-well part
  scalarvalueType gdwV = (n1 * n1 * (constV(1.0) - n1) * (constV(1.0) - n1) +
                          n2 * n2 * (constV(1.0) - n2) * (constV(1.0) - n2) +
                          n3 * n3 * (constV(1.0) - n3) * (constV(1.0) - n3) +
                          n4 * n4 * (constV(1.0) - n4) * (constV(1.0) - n4));
  // Interaction part
  scalarvalueType gintV =
    (alpha * (n1 * n1 * n2 * n2 + n1 * n1 * n3 * n3 + n1 * n1 * n4 * n4 +
              n2 * n2 * n3 * n3 + n2 * n2 * n4 * n4 + n3 * n3 * n4 * n4));
  // Combined function (g)
  scalarvalueType gV = (gdwV + gintV);

  // The homogenous free energy
  scalarvalueType f_chem = (constV(1.0) - hV) * faV + hV * fbV + wV * gV;

  // The gradient free energy
  scalarvalueType f_grad =
    constV(0.5) * KnV * (n1x * n1x + n2x * n2x + n3x * n3x + n4x * n4x) +
    constV(0.5) * KcV * cx * cx;

  // The total free energy
  scalarvalueType f_tot;
  f_tot = f_chem + f_grad;

  // --- Submitting the terms for the postprocessing expressions ---

  pp_variable_list.set_scalar_value_term_rhs(0, f_tot);
}