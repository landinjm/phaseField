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

void variableAttributeLoader::loadVariableAttributes(){
	// Variable 0 - the velocity vector
	set_variable_name				(0,"u");
	set_variable_type				(0,VECTOR);
	set_variable_equation_type		(0,EXPLICIT_TIME_DEPENDENT);

    set_dependencies_value_term_RHS(0, "u,grad(u),grad(P),grad(p_old),grad(divU)");
    set_dependencies_gradient_term_RHS(0, "grad(u)");

	// Variable 1 - pressure
	set_variable_name				(1,"P");
	set_variable_type				(1,SCALAR);
	set_variable_equation_type		(1,TIME_INDEPENDENT);

    set_dependencies_value_term_RHS(1, "grad(u)");
    set_dependencies_gradient_term_RHS(1, "grad(P),u,grad(u),grad(p_old),grad(divU),pi");
	set_dependencies_value_term_LHS(1, "");
    set_dependencies_gradient_term_LHS(1, "grad(change(P))");

	// Variable 2 - the previous pressure field
	set_variable_name				(2,"p_old");
	set_variable_type				(2,SCALAR);
	set_variable_equation_type		(2,EXPLICIT_TIME_DEPENDENT);

    set_dependencies_value_term_RHS(2, "P, p_old");
    set_dependencies_gradient_term_RHS(2, "");

	// Variable 3 - the mass conservation
	set_variable_name				(3,"divU");
	set_variable_type				(3,SCALAR);
	set_variable_equation_type		(3,EXPLICIT_TIME_DEPENDENT);

    set_dependencies_value_term_RHS(3, "");
    set_dependencies_gradient_term_RHS(3, "u");

	// Variable 4 - the projected pressure gradient
	set_variable_name				(4,"pi");
	set_variable_type				(4,VECTOR);
	set_variable_equation_type		(4,EXPLICIT_TIME_DEPENDENT);

    set_dependencies_value_term_RHS(4, "pi,grad(P)");
    set_dependencies_gradient_term_RHS(4, "");

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
void customPDE<dim,degree>::explicitEquationRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
				 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

	//Grab derivative model variable
	vectorvalueType u = variable_list.get_vector_value(0);
	vectorgradType ux = variable_list.get_vector_gradient(0);
	scalarvalueType P = variable_list.get_scalar_value(1);
	scalargradType Px = variable_list.get_scalar_gradient(1);
	scalarvalueType p_old = variable_list.get_scalar_value(2);
	scalargradType px_old = variable_list.get_scalar_gradient(2);
	scalargradType divUx = variable_list.get_scalar_gradient(3);
	vectorvalueType pi = variable_list.get_vector_value(4);

	//Initialize submission terms
	vectorvalueType eq_u = u;
	vectorgradType eqx_u;
	eqx_u = eqx_u*constV(0.0);
	scalarvalueType eq_p_old = p_old;
	scalargradType eqx_divU;
	eqx_divU = eqx_divU*constV(0.0);
	vectorvalueType eq_pi = pi;

	//Step one of the Chorin projection
	if(!ChorinSwitch){
		//Calculating the advection term
		vectorvalueType advecTerm;
		for(unsigned int i=0; i<dim; i++){
			for(unsigned int j=0; j<dim; j++){
				advecTerm[i] += u[j]*ux[i][j];
			}
		}

		//forcing term
		vectorvalueType forceTerm;
		for(unsigned int i=0; i<dim; i++){
			forceTerm[i] = 0.0;
			//if(i==1){forceTerm=-0.0;}
		}

		eq_u = u-constV(userInputs.dtValue)*(advecTerm-forceTerm+constV(alpha)*px_old);
		eqx_u = constV(-userInputs.dtValue/Re)*ux;
		eq_p_old = P;
		eqx_divU = -u;
	}

	//Step three of the Chorin projection
	if(ChorinSwitch == true){
		//Setting the expressions for the terms in the governing equations
		eq_u = u-constV(userInputs.dtValue)*(Px-constV(alpha)*px_old+constV(beta/Re)*divUx);
		eq_pi = Px;
	}

	//Submitting the terms for the governing equations
	variable_list.set_vector_value_term_RHS(0,eq_u);
	variable_list.set_vector_gradient_term_RHS(0,eqx_u);
	variable_list.set_scalar_value_term_RHS(2,eq_p_old);
	variable_list.set_scalar_gradient_term_RHS(3,eqx_divU);
	variable_list.set_vector_value_term_RHS(4,eq_pi);
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
void customPDE<dim,degree>::nonExplicitEquationRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
				 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

	//Grab derivative model variable
	vectorvalueType u = variable_list.get_vector_value(0);
	vectorgradType ux = variable_list.get_vector_gradient(0);
	scalargradType Px = variable_list.get_scalar_gradient(1);
	scalargradType px_old = variable_list.get_scalar_gradient(2);
	scalargradType divUx = variable_list.get_scalar_gradient(3);
	vectorvalueType pi = variable_list.get_vector_value(4);

	//Initialize submission terms
	scalarvalueType eq_P = constV(0.0);
	scalargradType eqx_P;
	eqx_P = eqx_P*constV(0.0);

	//Continuity equation
	for(unsigned int i=0; i<dim; i++){
		eq_P += -constV(1.0/userInputs.dtValue)*ux[i][i];
	}
	eqx_P = -Px;
	//pressure stabilization
	eqx_P += constV(delta/userInputs.dtValue)*(pi-Px);
	//incremental pressure-correction scheme
	eqx_P += constV(alpha)*px_old;
	//rotational incremental scheme
	eqx_P += -constV(beta/Re)*divUx;

	//pspg
	vectorvalueType advecTerm;
	for(unsigned int i=0; i<dim; i++){
		for(unsigned int j=0; j<dim; j++){
			advecTerm[i] += u[j]*ux[i][j];
		}
	}
	eqx_P += -tau*advecTerm;

	//Submitting the terms for the governing equations
	variable_list.set_scalar_value_term_RHS(1,eq_P);
	variable_list.set_scalar_gradient_term_RHS(1,eqx_P);
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
void customPDE<dim,degree>::equationLHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
		dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {
	
	// --- Getting the values and derivatives of the model variables ---
  	scalargradType DPx = variable_list.get_change_in_scalar_gradient(1);

	// --- Submitting the terms for the governing equations ---
	variable_list.set_scalar_gradient_term_LHS(1,constV(1.0+delta/userInputs.dtValue)*DPx);

}
