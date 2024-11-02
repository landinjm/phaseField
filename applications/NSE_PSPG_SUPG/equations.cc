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
  set_variable_name(0, "u_star");
  set_variable_type(0, VECTOR);
  set_variable_equation_type(0, TIME_INDEPENDENT);

  set_dependencies_value_term_RHS(0, "u_star, u_old, grad(u_old)");
  set_dependencies_gradient_term_RHS(0, "u_star, u_old, grad(u_old), hess(u_old)");
  set_dependencies_value_term_LHS(0, "change(u_star)");
  set_dependencies_gradient_term_LHS(0, "u_old, change(u_star)");

  // Variable 1
  set_variable_name(1, "u_old");
  set_variable_type(1, VECTOR);
  set_variable_equation_type(1, EXPLICIT_TIME_DEPENDENT);

  set_dependencies_value_term_RHS(1, "u_old, u");
  set_dependencies_gradient_term_RHS(1, "");

  // Variable 2
  set_variable_name(2, "p");
  set_variable_type(2, SCALAR);
  set_variable_equation_type(2, TIME_INDEPENDENT);

  set_dependencies_value_term_RHS(2, "grad(u_star)");
  set_dependencies_gradient_term_RHS(2,
                                     "grad(p), u_star, u_old, grad(u_old), hess(u_old)");
  set_dependencies_value_term_LHS(2, "");
  set_dependencies_gradient_term_LHS(2, "grad(change(p))");

  // Variable 3
  set_variable_name(3, "u");
  set_variable_type(3, VECTOR);
  set_variable_equation_type(3, TIME_INDEPENDENT);

  set_dependencies_value_term_RHS(3, "u, u_star grad(p)");
  set_dependencies_gradient_term_RHS(3, "u_star, u_old, grad(p)");
  set_dependencies_value_term_LHS(3, "change(u)");
  set_dependencies_gradient_term_LHS(3, "u_old, change(u)");
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
  vectorvalueType u = variable_list.get_vector_value(3);

  // Submitting the terms for the governing equations
  variable_list.set_vector_value_term_RHS(1, u);
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
  if (this->currentFieldIndex == 0)
    {
      vectorvalueType u_star  = variable_list.get_vector_value(0);
      vectorvalueType u_old   = variable_list.get_vector_value(1);
      vectorgradType  ux_old  = variable_list.get_vector_gradient(1);
      vectorhessType  uxx_old = variable_list.get_vector_hessian(1);

      scalarvalueType stabilization_parameter =
        compute_stabilization_parameter(u_old, element_volume);

      vectorvalueType advection_term;
      advection_term = constV(0.0) * advection_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              advection_term[i] += u_old[j] * ux_old[i][j];
            }
        }

      vectorvalueType residual = u_old - u_star - dt * advection_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          residual += dt * nu * uxx_old[i][i];
        }

      vectorgradType SUPG_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              SUPG_term[i][j] = residual[i] * u_old[j];
            }
        }
      SUPG_term *= stabilization_parameter;

      vectorvalueType eq_u_star  = u_old - u_star - dt * advection_term;
      vectorgradType  eqx_u_star = -dt * ux_old * nu + SUPG_term;

      variable_list.set_vector_value_term_RHS(0, eq_u_star);
      variable_list.set_vector_gradient_term_RHS(0, eqx_u_star);
    }
  else if (this->currentFieldIndex == 2)
    {
      vectorvalueType u_star  = variable_list.get_vector_value(0);
      vectorgradType  ux_star = variable_list.get_vector_gradient(0);
      vectorvalueType u_old   = variable_list.get_vector_value(1);
      vectorgradType  ux_old  = variable_list.get_vector_gradient(1);
      vectorhessType  uxx_old = variable_list.get_vector_hessian(1);
      scalargradType  px      = variable_list.get_scalar_gradient(2);

      scalarvalueType stabilization_parameter =
        compute_stabilization_parameter(u_star, element_volume);

      scalarvalueType eq_p = constV(0.0);
      for (unsigned int i = 0; i < dim; i++)
        {
          eq_p -= ux_star[i][i] / (dt + stabilization_parameter);
        }
      scalargradType eqx_p = -px;

      vectorvalueType advection_term;
      advection_term = constV(0.0) * advection_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              advection_term[i] += u_old[j] * ux_old[i][j];
            }
        }

      scalargradType residual = (u_star - u_old) / dt + advection_term;
      for (unsigned int i = 0; i < dim; i++)
        {
          residual -= nu * uxx_old[i][i];
        }
      eqx_p -= residual * stabilization_parameter / (dt + stabilization_parameter);

      variable_list.set_scalar_value_term_RHS(2, eq_p);
      variable_list.set_scalar_gradient_term_RHS(2, eqx_p);
    }
  else if (this->currentFieldIndex == 3)
    {
      vectorvalueType u_star = variable_list.get_vector_value(0);
      vectorvalueType u_old  = variable_list.get_vector_value(1);
      scalargradType  px     = variable_list.get_scalar_gradient(2);
      vectorvalueType u      = variable_list.get_vector_value(3);

      scalarvalueType stabilization_parameter =
        compute_stabilization_parameter(u_old, element_volume);

      vectorvalueType residual = u_star - u - dt * px;

      vectorgradType eq_ux;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              eq_ux[i][j] = residual[i] * u_old[j];
            }
        }
      eq_ux *= stabilization_parameter;

      vectorvalueType eq_u = u_star - u - dt * px;

      variable_list.set_vector_value_term_RHS(3, eq_u);
      variable_list.set_vector_gradient_term_RHS(3, eq_ux);
    }
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
  if (this->currentFieldIndex == 0)
    {
      vectorvalueType D_u_star = variable_list.get_change_in_vector_value(0);
      vectorvalueType u_old    = variable_list.get_vector_value(1);

      scalarvalueType stabilization_parameter =
        compute_stabilization_parameter(u_old, element_volume);

      vectorgradType eq_Dx_u_star;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              eq_Dx_u_star[i][j] = D_u_star[i] * u_old[j];
            }
        }
      eq_Dx_u_star *= stabilization_parameter;

      variable_list.set_vector_value_term_LHS(0, D_u_star);
      variable_list.set_vector_gradient_term_LHS(0, eq_Dx_u_star);
    }
  else if (this->currentFieldIndex == 2)
    {
      scalargradType D_px = variable_list.get_change_in_scalar_gradient(2);

      variable_list.set_scalar_gradient_term_LHS(2, D_px);
    }
  else if (this->currentFieldIndex == 3)
    {
      vectorvalueType u_old = variable_list.get_vector_value(1);
      vectorvalueType D_u   = variable_list.get_change_in_vector_value(3);

      scalarvalueType stabilization_parameter =
        compute_stabilization_parameter(u_old, element_volume);

      vectorgradType eq_Dx_u;
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = 0; j < dim; j++)
            {
              eq_Dx_u[i][j] = D_u[i] * u_old[j];
            }
        }
      eq_Dx_u *= stabilization_parameter;

      variable_list.set_vector_value_term_LHS(3, D_u);
      variable_list.set_vector_gradient_term_LHS(3, eq_Dx_u);
    }
}
