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
  set_variable_name(0, "error_squared");
  set_variable_type(0, Scalar);

  set_dependencies_value_term_rhs(0, "n, grad(n)");
  set_dependencies_gradient_term_rhs(0, "");

  set_output_integral(0, true);

  // Variable 1
  set_variable_name(1, "f_tot");
  set_variable_type(1, Scalar);

  set_dependencies_value_term_rhs(1, "n, grad(n)");
  set_dependencies_gradient_term_rhs(1, "");

  set_output_integral(1, true);

  // Variable 1
  set_variable_name(2, "src");
  set_variable_type(2, Scalar);

  set_dependencies_value_term_rhs(2, "n, grad(n)");
  set_dependencies_gradient_term_rhs(2, "");

  set_output_integral(2, true);

  // Variable 1
  set_variable_name(3, "n_sol");
  set_variable_type(3, Scalar);

  set_dependencies_value_term_rhs(3, "n, grad(n)");
  set_dependencies_gradient_term_rhs(3, "");

  set_output_integral(3, true);
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

  // The order parameter and its derivatives
  scalarvalueType n  = variable_list.template get_value<ScalarValue>(0);
  scalargradType  nx = variable_list.template get_gradient<ScalarGrad>(0);

  // --- Setting the expressions for the terms in the postprocessing expressions
  // ---

  scalarvalueType f_tot = constV(0.0);

  // The homogenous free energy
  scalarvalueType f_chem = (n * n * n * n - 2.0 * n * n * n + n * n);

  // The gradient free energy
  scalarvalueType f_grad = constV(0.0);

  for (int i = 0; i < dim; i++)
    {
      for (int j = 0; j < dim; j++)
        {
          f_grad += constV(0.5 * kappa) * nx[i] * nx[j];
        }
    }

  // The total free energy
  f_tot = f_chem + f_grad;

  scalarvalueType source_term;
  scalarvalueType n_sol;

  scalarvalueType alpha = 0.25 + A1 * this->currentTime * std::sin(B1 * q_point_loc(0)) +
                          A2 * std::sin(B2 * q_point_loc(0) + C2 * this->currentTime);
  scalarvalueType alpha_t =
    A1 * std::sin(B1 * q_point_loc(0)) +
    A2 * C2 * std::cos(B2 * q_point_loc(0) + C2 * this->currentTime);
  scalarvalueType alpha_y =
    A1 * B1 * this->currentTime * std::cos(B1 * q_point_loc(0)) +
    A2 * B2 * std::cos(B2 * q_point_loc(0) + C2 * this->currentTime);
  scalarvalueType alpha_yy =
    -A1 * B1 * B1 * this->currentTime * std::sin(B1 * q_point_loc(0)) -
    A2 * B2 * B2 * std::sin(B2 * q_point_loc(0) + C2 * this->currentTime);

  for (unsigned i = 0; i < n.size(); i++)
    {
      source_term[i] =
        (-2.0 * std::sqrt(kappa) *
           std::tanh((q_point_loc(1)[i] - alpha[i]) / std::sqrt(2.0 * kappa)) *
           (alpha_y[i] * alpha_y[i]) +
         std::sqrt(2.0) * (alpha_t[i] - kappa * alpha_yy[i])) /
        (4.0 * std::sqrt(kappa)) /
        Utilities::fixed_power<2>(
          std::cosh((q_point_loc(1)[i] - alpha[i]) / std::sqrt(2.0 * kappa)));

      n_sol[i] =
        0.5 * (1.0 - std::tanh((q_point_loc(1)[i] - alpha[i]) / std::sqrt(2.0 * kappa)));
    }

  scalarvalueType error = (n_sol - n) * (n_sol - n);

  // --- Submitting the terms for the postprocessing expressions ---

  pp_variable_list.set_scalar_value_term_rhs(0, error);
  pp_variable_list.set_scalar_value_term_rhs(1, f_tot);
  pp_variable_list.set_scalar_value_term_rhs(2, source_term);
  pp_variable_list.set_scalar_value_term_rhs(3, n_sol);
}