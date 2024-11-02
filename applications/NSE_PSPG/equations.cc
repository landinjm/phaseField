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
  set_variable_name(0, "u");
  set_variable_type(0, VECTOR);
  set_variable_equation_type(0, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(0, "u, grad(p), grad(u)");
  set_dependencies_gradient_term_RHS(0, "grad(u)");

  // Variable 1
  set_variable_name(1, "p");
  set_variable_type(1, SCALAR);
  set_variable_equation_type(1, TIME_INDEPENDENT);

  set_dependencies_value_term_RHS(1, "grad(u)");
  set_dependencies_gradient_term_RHS(1, "grad(p), u, u_old, grad(u), hess(u)");
  set_dependencies_value_term_LHS(1, "");
  set_dependencies_gradient_term_LHS(1, "grad(change(p))");

  // Variable 2
  set_variable_name(2, "u_old");
  set_variable_type(2, VECTOR);
  set_variable_equation_type(2, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(2, "u_old, u");
  set_dependencies_gradient_term_RHS(2, "");
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
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{
  // Grab model variables
  vectorvalueType u  = variable_list.get_vector_value(0);
  vectorgradType  ux = variable_list.get_vector_gradient(0);
  scalargradType  px = variable_list.get_scalar_gradient(1);

  // Initialize submission terms
  vectorvalueType eq_u = u;
  vectorgradType  eqx_u;
  eqx_u = eqx_u * constV(0.0);

  // Step one of the Chorin projection
  if (!fractional_pressure_update_step)
    {
      vectorvalueType advection_term;
      advection_term = constV(0.0) * advection_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              advection_term[i] += u[j] * ux[i][j];
            }
        }

      eq_u  = u - dt * advection_term;
      eqx_u = -dt * ux * nu;
    }

  // Step three of the Chorin projection
  else
    {
      eq_u = u - dt * px;
    }

  // Submitting the terms for the governing equations
  variable_list.set_vector_value_term_RHS(0, eq_u);
  variable_list.set_vector_gradient_term_RHS(0, eqx_u);
  variable_list.set_vector_value_term_RHS(2, u);
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
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{
  // Grab model variables
  vectorvalueType u     = variable_list.get_vector_value(0);
  vectorgradType  ux    = variable_list.get_vector_gradient(0);
  vectorhessType  uxx   = variable_list.get_vector_hessian(0);
  scalargradType  px    = variable_list.get_scalar_gradient(1);
  vectorvalueType u_old = variable_list.get_vector_value(2);

  scalarvalueType stabilization_parameter =
    compute_stabilization_parameter(u, element_volume);

  // Continuity equation
  scalarvalueType eq_p = constV(0.0);
  for (unsigned int i = 0; i < dim; i++)
    {
      eq_p -= ux[i][i] / (dt + stabilization_parameter);
    }
  scalargradType eqx_p = -px;

  vectorvalueType advection_term;
  advection_term = constV(0.0) * advection_term;
  for (unsigned int i = 0; i < dim; i++)
    {
      for (unsigned int j = 0; j < dim; j++)
        {
          advection_term[i] += u[j] * ux[i][j];
        }
    }

  // Residual
  scalargradType residual = (u - u_old) / dt;
  for (unsigned int i = 0; i < dim; i++)
    {
      residual -= nu * uxx[i][i] + advection_term;
    }
  eqx_p -= residual * stabilization_parameter / (dt + stabilization_parameter);

  // Submitting the terms for the governing equations
  variable_list.set_scalar_value_term_RHS(1, eq_p);
  variable_list.set_scalar_gradient_term_RHS(1, eqx_p);
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
  [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>> &variable_list,
  [[maybe_unused]] const Point<dim, VectorizedArray<double>>                q_point_loc,
  [[maybe_unused]] const VectorizedArray<double> element_volume) const
{
  scalargradType Dpx = variable_list.get_change_in_scalar_gradient(1);

  variable_list.set_scalar_gradient_term_LHS(1, Dpx);
}