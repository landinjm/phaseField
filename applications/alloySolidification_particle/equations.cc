// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
// This function sets attributes for each variable/equation in the app. The
// attributes are set via standardized function calls. The first parameter for
// each function call is the variable index (starting at zero). The first set of
// variable/equation attributes are the variable name (any string), the variable
// type (SCALAR/VECTOR), and the equation type (EXPLICIT_TIME_DEPENDENT/
// TIME_INDEPENDENT/AUXILIARY). The next set of attributes describe the
// dependencies for the governing equation on the values and derivatives of the
// other variables for the value term and gradient term of the RHS and the LHS.
// The final pair of attributes determine whether a variable represents a field
// that can nucleate and whether the value of the field is needed for nucleation
// rate calculations.

void
variableAttributeLoader::loadVariableAttributes()
{
  // Variable 0
  set_variable_name(0, "U");
  set_variable_type(0, SCALAR);
  set_variable_equation_type(0, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(0, "U,xi,phi,grad(phi),grad(U)");
  set_dependencies_gradient_term_RHS(0, "U,grad(U),grad(phi),phi,xi");

  // Variable 1
  set_variable_name(1, "phi");
  set_variable_type(1, SCALAR);
  set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(1, "phi,U,xi");
  set_dependencies_gradient_term_RHS(1, "");

  // Variable 2
  set_variable_name(2, "xi");
  set_variable_type(2, SCALAR);
  set_variable_equation_type(2, AUXILIARY);

  set_dependencies_value_term_RHS(2, "phi,U,grad(phi)");
  set_dependencies_gradient_term_RHS(2, "grad(phi)");

  // Variable 3
  set_variable_name(3, "psi_level_set");
  set_variable_type(3, SCALAR);
  set_variable_equation_type(3, TIME_INDEPENDENT);

  set_dependencies_value_term_RHS(
    3,
    "psi_level_set_old, psi_level_set, grad(psi_level_set)");
  set_dependencies_gradient_term_RHS(
    3,
    "psi_level_set_old, psi_level_set, grad(psi_level_set)");
  set_dependencies_value_term_LHS(3,
                                  "change(psi_level_set), grad(change(psi_level_set))");
  set_dependencies_gradient_term_LHS(
    3,
    "change(psi_level_set), grad(change(psi_level_set))");

  // Variable 4
  set_variable_name(4, "psi_level_set_old");
  set_variable_type(4, SCALAR);
  set_variable_equation_type(4, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(4, "psi_level_set");
  set_dependencies_gradient_term_RHS(4, "");
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
customPDE<dim, degree>::explicitEquationRHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
  dealii::VectorizedArray<double>                                  element_volume) const
{
  // --- Getting the values and derivatives of the model variables ---

  // The dimensionless solute supersaturation and its derivatives
  scalarvalueType U  = variable_list.get_scalar_value(0);
  scalargradType  Ux = variable_list.get_scalar_gradient(0);

  // The order parameter and its derivatives
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // The auxiliary parameter and its derivatives
  scalarvalueType xi = variable_list.get_scalar_value(2);

  // The level-set field of the particle
  scalarvalueType psi_level_set = variable_list.get_scalar_value(3);

  // --- Setting the expressions for the terms in the governing equations ---

  // Calculation of interface normal vector
  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = phix / (normgradn + constV(regval));

  // The cosine of theta
  scalarvalueType cth = normal[0];
  // The sine of theta
  scalarvalueType sth = normal[1];
  // The cosine of 4 theta
  scalarvalueType c4th =
    sth * sth * sth * sth + cth * cth * cth * cth - constV(6.0) * sth * sth * cth * cth;

  // Anisotropic term
  scalarvalueType a_n;
  a_n = (constV(1.0) + constV(epsilon) * c4th);

  // coeffcient before phi
  scalarvalueType tau_phi = (constV(1.0) + constV(1.0 - k) * U) * a_n * a_n;

  // coeffcient before U
  scalarvalueType tau_U =
    (constV(1.0 + k) / constV(2.0) - constV(1.0 - k) * phi / constV(2.0));

  // Antitrapping term
  scalargradType j_at;
  j_at = constV(1.0 / (2.0 * sqrt(2.0))) * (constV(1.0) + constV(1.0 - k) * U) *
         (xi / tau_phi) * normal;

  // grad_phi and grad_U dot product term
  scalarvalueType val_term1 = constV(userInputs.dtValue) *
                              (constV(1.0) + constV(1.0 - k) * U) * xi /
                              (constV(2.0) * tau_phi * tau_U);
  scalarvalueType val_term2 =
    constV(userInputs.dtValue) *
    (constV((1.0 - k) / 2.0) / (tau_U * tau_U) *
     (phix[0] * (Dtilde * ((constV(1.0) - phi) / constV(2.0)) * Ux[0] + j_at[0]) +
      phix[1] * (Dtilde * ((constV(1.0) - phi) / constV(2.0)) * Ux[1] + j_at[1])));

  // Define required equations
  scalarvalueType eq_U = (U + val_term1 - val_term2);

  scalargradType eqx_U =
    (constV(-1.0) * constV(userInputs.dtValue) *
     (Dtilde * ((constV(1.0) - phi) / constV(2.0)) * Ux + j_at) / tau_U);

  scalarvalueType eq_phi = (phi + constV(userInputs.dtValue) * xi / tau_phi);

  // --- Submitting the terms for the governing equations ---

  // Terms for the equation to evolve the concentration
  variable_list.set_scalar_value_term_RHS(0, eq_U);
  variable_list.set_scalar_gradient_term_RHS(0, eqx_U);

  // Terms for the equation to evolve the order parameter
  variable_list.set_scalar_value_term_RHS(1, eq_phi);

  // Terms for the equation to evolve the particle level set field
  variable_list.set_scalar_value_term_RHS(4, psi_level_set);
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
customPDE<dim, degree>::nonExplicitEquationRHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
  dealii::VectorizedArray<double>                                  element_volume) const
{
  // --- Getting the values and derivatives of the model variables ---

  // The temperature and its derivatives
  scalarvalueType U = variable_list.get_scalar_value(0);

  // The order parameter and its derivatives
  scalarvalueType phi  = variable_list.get_scalar_value(1);
  scalargradType  phix = variable_list.get_scalar_gradient(1);

  // The level-set field of the particle
  scalarvalueType psi_level_set     = variable_list.get_scalar_value(3);
  scalargradType  psix_level_set    = variable_list.get_scalar_gradient(3);
  scalarvalueType psi_level_set_old = variable_list.get_scalar_value(4);

  // --- Setting the expressions for the terms in the governing equations ---

  // Calculation of interface normal vector
  scalarvalueType normgradn = std::sqrt(phix.norm_square());
  scalargradType  normal    = phix / (normgradn + constV(regval));

  // The cosine of theta
  scalarvalueType cth = normal[0];
  // The sine of theta
  scalarvalueType sth = normal[1];

  // The cosine of 4 theta
  scalarvalueType c4th =
    sth * sth * sth * sth + cth * cth * cth * cth - constV(6.0) * sth * sth * cth * cth;
  // The sine of 4 theta
  scalarvalueType s4th =
    constV(4.0) * sth * cth * cth * cth - constV(4.0) * sth * sth * sth * cth;

  // Anisotropic term
  scalarvalueType a_n;
  a_n = (constV(1.0) + constV(epsilon) * c4th);

  // gradient energy coefficient, its derivative and square
  scalarvalueType a_d;
  a_d = constV(-4.0) * constV(epsilon) * s4th;

  // dimensionless temperature changes
  scalarvalueType y   = q_point_loc[1];            // The y-component
  scalarvalueType t_n = constV(this->currentTime); // The time
  scalarvalueType tep = ((y - constV(y0) - Vtilde * t_n) / ltilde);

  // The anisotropy term that enters in to the equation for xi
  scalargradType aniso;
  aniso[0] = a_n * a_n * phix[0] - a_n * a_d * phix[1];
  aniso[1] = a_n * a_n * phix[1] + a_n * a_d * phix[0];

  // Define the terms in the equations
  scalarvalueType eq_xi = phi - phi * phi * phi -
                          constV(lamda) * (constV(1.0) - phi * phi) *
                            (constV(1.0) - phi * phi) * (U + tep + constV(U_off));

  scalargradType eqx_xi = (-aniso);

  /*
   * Particle advection
   */

  // Particle velocity
  scalargradType velocity;
  velocity[0] = constV(0.0);
  velocity[1] = constV(1.0);

  // Stabilization parameter
  scalarvalueType u_l2norm = 1.0e-12 + velocity.norm_square();
  scalarvalueType h = std::sqrt(element_volume) * constV(std::sqrt(4.0 / M_PI) / degree);
  scalarvalueType stabilization_parameter =
    constV(1.0) / std::sqrt(constV(dealii::Utilities::fixed_power<2>(sdt)) +
                            constV(4.0) * u_l2norm / h / h);

  // Backwards Euler Advection
  scalarvalueType residual = (psi_level_set_old - psi_level_set -
                              constV(userInputs.dtValue) * velocity * psix_level_set);

  scalarvalueType eq_psi_level_set  = residual;
  scalargradType  eqx_psi_level_set = residual * stabilization_parameter * velocity;

  // --- Submitting the terms for the governing equations ---

  variable_list.set_scalar_value_term_RHS(2, eq_xi);
  variable_list.set_scalar_gradient_term_RHS(2, eqx_xi);
  variable_list.set_scalar_value_term_RHS(3, eq_psi_level_set);
  variable_list.set_scalar_gradient_term_RHS(3, eqx_psi_level_set);
}

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
customPDE<dim, degree>::equationLHS(
  variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
  dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
  dealii::VectorizedArray<double>                                  element_volume) const
{
  // The level-set field of the particle
  scalarvalueType change_psi_level_set  = variable_list.get_change_in_scalar_value(3);
  scalargradType  change_psix_level_set = variable_list.get_change_in_scalar_gradient(3);

  // Particle velocity
  scalargradType velocity;
  velocity[0] = constV(0.0);
  velocity[1] = constV(1.0);

  // Stabilization parameter
  scalarvalueType u_l2norm = 1.0e-12 + velocity.norm_square();
  scalarvalueType h = std::sqrt(element_volume) * constV(std::sqrt(4.0 / M_PI) / degree);
  scalarvalueType stabilization_parameter =
    constV(1.0) / std::sqrt(constV(dealii::Utilities::fixed_power<2>(sdt)) +
                            constV(4.0) * u_l2norm / h / h);

  scalarvalueType residual = (change_psi_level_set + constV(userInputs.dtValue) *
                                                       velocity * change_psix_level_set);

  scalarvalueType eq_psi_level_set  = residual;
  scalargradType  eqx_psi_level_set = residual * stabilization_parameter * velocity;

  variable_list.set_scalar_value_term_LHS(3, eq_psi_level_set);
  variable_list.set_scalar_gradient_term_LHS(3, eqx_psi_level_set);
}
