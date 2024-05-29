// =============================================================================================
// loadPostProcessorVariableAttributes: Set the attributes of the postprocessing variables
// =============================================================================================
// This function is analogous to 'loadVariableAttributes' in 'equations.h', but for
// the postprocessing expressions. It sets the attributes for each postprocessing
// expression, including its name, whether it is a vector or scalar (only scalars are
// supported at present), its dependencies on other variables and their derivatives,
// and whether to calculate an integral of the postprocessed quantity over the entire
// domain. Note: this function is not a member of customPDE.

void variableAttributeLoader::loadPostProcessorVariableAttributes(){

	// Variable 0
	set_variable_name				(0,"l2normPressure");
	set_variable_type				(0,SCALAR);

	set_dependencies_value_term_RHS(0, "P");
	set_dependencies_gradient_term_RHS(0, "");

	set_output_integral         	(0,true);

	// Variable 1
	set_variable_name				(1,"l2normXvelocity");
	set_variable_type				(1,SCALAR);

	set_dependencies_value_term_RHS(1, "u");
	set_dependencies_gradient_term_RHS(1, "");

	set_output_integral         	(1,true);

	// Variable 2
	set_variable_name				(2,"l2normYvelocity");
	set_variable_type				(2,SCALAR);

	set_dependencies_value_term_RHS(2, "u");
	set_dependencies_gradient_term_RHS(2, "");

	set_output_integral         	(2,true);

}

// =============================================================================================
// postProcessedFields: Set the postprocessing expressions
// =============================================================================================
// This function is analogous to 'explicitEquationRHS' and 'nonExplicitEquationRHS' in
// equations.h. It takes in "variable_list" and "q_point_loc" as inputs and outputs two terms in
// the expression for the postprocessing variable -- one proportional to the test
// function and one proportional to the gradient of the test function. The index for
// each variable in this list corresponds to the index given at the top of this file (for
// submitting the terms) and the index in 'equations.h' for assigning the values/derivatives of
// the primary variables.

template <int dim,int degree>
void customPDE<dim,degree>::postProcessedFields(const variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
				variableContainer<dim,degree,dealii::VectorizedArray<double> > & pp_variable_list,
												const dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

	vectorvalueType u = variable_list.get_vector_value(0);
	scalarvalueType P = variable_list.get_scalar_value(1);

	scalarvalueType adjustedx = q_point_loc[0]-constV(0.5);
	scalarvalueType adjustedy = q_point_loc[1]-constV(0.5);

	scalarvalueType lambda = constV(0.5*(Re-std::sqrt(Re*Re+16.0*M_PI*M_PI)));
	scalarvalueType xponent = std::exp(lambda*adjustedx);

	scalarvalueType pressure = constV(0.5) - constV(0.5)*xponent*xponent;

	scalarvalueType xvelocity = constV(1.0) - xponent*std::cos(constV(2.0*M_PI)*adjustedy);
	scalarvalueType yvelocity = lambda/constV(2.0*M_PI)*xponent*std::sin(constV(2.0*M_PI)*adjustedy);

	//Note for some reason PRISMS-PF freaks out when the dirichlet is set to something other
	//than 0. This parameter allows for tuning based on analytical pressure solution.
	scalarvalueType maxX = constV(userInputs.domain_size[0]-0.5);
	scalarvalueType temp = std::exp(lambda*maxX);
	pressure -= constV(0.5) - constV(0.5)*temp*temp;

	//Calculating l2norm
	pressure = std::pow(P-pressure,2.0);
	xvelocity = std::pow(u[0]-xvelocity,2.0);
	yvelocity = std::pow(u[1]-yvelocity,2.0);

	pp_variable_list.set_scalar_value_term_RHS(0,pressure);
	pp_variable_list.set_scalar_value_term_RHS(1,xvelocity);
	pp_variable_list.set_scalar_value_term_RHS(2,yvelocity);

}
