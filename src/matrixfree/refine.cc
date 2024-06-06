//mesh refinement methods for MatrixFreePDE class

#include "../../include/matrixFreePDE.h"
#include <deal.II/distributed/grid_refinement.h>

//default implementation of adaptive mesh refinement
template <int dim, int degree>
void MatrixFreePDE<dim,degree>::adaptiveRefine(unsigned int currentIncrement){
if (userInputs.h_adaptivity == true){
	if ( (currentIncrement == 0) ){
		computing_timer.enter_subsection("matrixFreePDE: AMR");
		unsigned int numDoF_preremesh = totalDOFs;
		for (unsigned int remesh_index=0; remesh_index < (userInputs.max_refinement_level-userInputs.min_refinement_level); remesh_index++){

			adaptiveRefineCriterion();
			refineGrid();
			reinit();

			// If the mesh hasn't changed from the previous cycle, stop remeshing
			if (totalDOFs == numDoF_preremesh) break;
			numDoF_preremesh = totalDOFs;
		}
		computing_timer.leave_subsection("matrixFreePDE: AMR");
	}
	else if ( (currentIncrement%userInputs.skip_remeshing_steps==0) ){

		computing_timer.enter_subsection("matrixFreePDE: AMR");

		// Apply constraints before remeshing
		for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
			constraintsDirichletSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
			constraintsOtherSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
			solutionSet[fieldIndex]->update_ghost_values();
		}
		adaptiveRefineCriterion();
		refineGrid();
		reinit();
		computing_timer.leave_subsection("matrixFreePDE: AMR");
	}
}
}

//default implementation of adaptive mesh criterion
template <int dim, int degree>
void MatrixFreePDE<dim,degree>::adaptiveRefineCriterion(){
// Old code to implement a Kelly error estimator
//Kelly error estimation criterion
//estimate cell wise errors for mesh refinement
//#if hAdaptivity==true
//#ifdef adaptivityType
//#if adaptivityType=="KELLY"
//  Vector<float> estimated_error_per_cell (triangulation.n_locally_owned_active_cells());
//  KellyErrorEstimator<dim>::estimate (*dofHandlersSet_nonconst[refinementDOF],
//				      QGaussLobatto<dim-1>(degree+1),
//				      typename FunctionMap<dim>::type(),
//				      *solutionSet[refinementDOF],
//				      estimated_error_per_cell,
//				      ComponentMask(),
//				      0,
//				      1,
//				      triangulation.locally_owned_subdomain());
//  //flag cells for refinement
//  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
//									    estimated_error_per_cell,
//									    topRefineFraction,
//									    bottomCoarsenFraction);
//#endif
//#endif
//#endif

{
std::vector<std::vector<double> > valuesV;
std::vector<std::vector<double> > gradientsV;

QGaussLobatto<dim>  quadrature(degree+1);
const unsigned int num_quad_points = quadrature.size();

// Set the correct update flags
//This ought to be separated based on the field type (SCALAR/VECTOR)
bool need_value = false;
bool need_gradient = false;
for (unsigned int field_index=0; field_index<userInputs.refinement_criteria.size(); field_index++){
    if (userInputs.refinement_criteria[field_index].criterion_type == VALUE || userInputs.refinement_criteria[field_index].criterion_type == VALUE_AND_GRADIENT){
        need_value = true;
    }
    else if (userInputs.refinement_criteria[field_index].criterion_type == GRADIENT || userInputs.refinement_criteria[field_index].criterion_type == VALUE_AND_GRADIENT){
        need_gradient = true;
    }
}
dealii::UpdateFlags update_flags;
if (need_value && !need_gradient){
    update_flags = update_values;
}
else if (!need_value && need_gradient){
    update_flags = update_gradients;
}
else {
    update_flags = update_values | update_gradients;
}

//Find the indices of first occuring scalar & vector field in the refinement criterion
unsigned int scalarField = 0;
bool foundScalar = false;
unsigned int vectorField = 0;
bool foundVector = false;
for(auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); it++){
	if(fields[it->variable_index].type==SCALAR && !foundScalar){
		scalarField = it->variable_index;
		foundScalar = true;
	}
	else if(fields[it->variable_index].type==VECTOR && !foundVector){
		vectorField = it->variable_index;
		foundVector = true;
	}
}

pcout << "Found Scalar Refinement Criteria = " << std::boolalpha << foundScalar << std::endl;
pcout << "Found Vector Refinement Criteria = " << std::boolalpha << foundVector << std::endl;

if(foundScalar){
	FEValues<dim> fe_values (*FESet[scalarField], quadrature, update_flags);

	std::vector<double> values(num_quad_points);
	std::vector<double> gradient_magnitudes(num_quad_points);
	std::vector<dealii::Tensor<1,dim,double> > gradients(num_quad_points);

	typename DoFHandler<dim>::active_cell_iterator cell = dofHandlersSet_nonconst[scalarField]->begin_active(), endc = dofHandlersSet_nonconst[scalarField]->end();

	typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell = triangulation.begin_active();

	for (;cell!=endc; ++cell){
		if (cell->is_locally_owned()){
			fe_values.reinit (cell);

			for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); it++){
				//Push back the refinement values. If we have the wrong type (e.g., VECTOR) push back an empty
				//element so the next loop works.
				if (need_value){
					if (fields[it->variable_index].type!=SCALAR){
						valuesV.emplace_back();
					}
					else{
						fe_values.get_function_values(*solutionSet[it->variable_index], values);
						valuesV.push_back(values);
					}
				}
				if (need_gradient){
					if (fields[it->variable_index].type!=SCALAR){
						gradientsV.emplace_back();
					}
					else{
						fe_values.get_function_gradients(*solutionSet[it->variable_index], gradients);

						for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
							gradient_magnitudes.at(q_point) = gradients.at(q_point).norm();
						}

						gradientsV.push_back(gradient_magnitudes);
					}
				}
			}

			bool mark_refine = false;

			for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
				for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); it++){
					if (fields[it->variable_index].type!=SCALAR){
						continue;
					}
					//Get index of iterator
					std::size_t index = std::distance(userInputs.refinement_criteria.begin(), it);

					//Mark refinement based on quadrature value and user criterion
					if (it->criterion_type == VALUE || it->criterion_type == VALUE_AND_GRADIENT){
						if ((valuesV[index][q_point]>it->value_lower_bound) && (valuesV[index][q_point]<it->value_upper_bound)){
							mark_refine = true;
							break;
						}
					}
					if (it->criterion_type == GRADIENT || it->criterion_type == VALUE_AND_GRADIENT){
						if (gradientsV[index][q_point]>it->gradient_lower_bound){
							mark_refine = true;
							break;
						}
					}
				}
			}
			valuesV.clear();
			gradientsV.clear();

			//limit the maximal and minimal refinement depth of the mesh
			unsigned int current_level = t_cell->level();

			if ( (mark_refine && current_level < userInputs.max_refinement_level) ){
				cell->clear_coarsen_flag();
				cell->set_refine_flag();
			}
			else if (!mark_refine && current_level > userInputs.min_refinement_level && !cell->refine_flag_set()) {
				cell->set_coarsen_flag();
			}

		}
		++t_cell;
	}
}

if(foundVector){
	FEValues<dim> fe_values (*FESet[vectorField], quadrature, update_flags);

	std::vector<double> value_magnitudes(num_quad_points);
	std::vector<dealii::Vector<double>> values(num_quad_points, dealii::Vector<double>(dim));
	std::vector<double> gradient_magnitudes(num_quad_points);
	dealii::Vector<double> gradient_magnitude_components(dim);
	std::vector<std::vector<dealii::Tensor<1,dim,double>> > gradients(num_quad_points, std::vector<dealii::Tensor<1,dim,double>>(dim));

	typename DoFHandler<dim>::active_cell_iterator cell = dofHandlersSet_nonconst[vectorField]->begin_active(), endc = dofHandlersSet_nonconst[vectorField]->end();

	typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell = triangulation.begin_active();

	for (;cell!=endc; ++cell){
		if (cell->is_locally_owned()){
			fe_values.reinit (cell);

			for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); it++){
				//Push back the refinement values. If we have the wrong type (e.g., VECTOR) push back an empty
				//element so the next loop works.
				if (need_value){
					if (fields[it->variable_index].type!=VECTOR){
						valuesV.emplace_back();
					}
					else{
						fe_values.get_function_values(*solutionSet[it->variable_index], values);
						for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
							value_magnitudes.at(q_point) = values.at(q_point).l2_norm();
						}
						valuesV.push_back(value_magnitudes);
					}
				}
				if (need_gradient){
					if (fields[it->variable_index].type!=VECTOR){
						gradientsV.emplace_back();
					}
					else{
						fe_values.get_function_gradients(*solutionSet[it->variable_index], gradients);

						for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
							for (unsigned int d = 0; d<dim; ++d){
								gradient_magnitude_components[d] = gradients.at(q_point).at(d).norm();
							}
							gradient_magnitudes.at(q_point) = gradient_magnitude_components.l2_norm();
						}

						gradientsV.push_back(gradient_magnitudes);
					}
				}
			}

			bool mark_refine = false;

			for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
				for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); it++){
					if (fields[it->variable_index].type!=VECTOR){
						continue;
					}
					//Get index of iterator
					std::size_t index = std::distance(userInputs.refinement_criteria.begin(), it);

					//Mark refinement based on quadrature value and user criterion
					if (it->criterion_type == VALUE || it->criterion_type == VALUE_AND_GRADIENT){
						if ((valuesV[index][q_point]>it->value_lower_bound) && (valuesV[index][q_point]<it->value_upper_bound)){
							mark_refine = true;
							break;
						}
					}
					if (it->criterion_type == GRADIENT || it->criterion_type == VALUE_AND_GRADIENT){
						if (gradientsV[index][q_point]>it->gradient_lower_bound){
							mark_refine = true;
							break;
						}
					}
				}
			}
			valuesV.clear();
			gradientsV.clear();

			//limit the maximal and minimal refinement depth of the mesh
			unsigned int current_level = t_cell->level();

			if ( (mark_refine && current_level < userInputs.max_refinement_level) ){
				cell->clear_coarsen_flag();
				cell->set_refine_flag();
			}
			else if (!mark_refine && current_level > userInputs.min_refinement_level && !cell->refine_flag_set()) {
				cell->set_coarsen_flag();
			}

		}
		++t_cell;
	}
}
}

}


//refine grid method
template <int dim, int degree>
void MatrixFreePDE<dim,degree>::refineGrid (){

//prepare and refine
triangulation.prepare_coarsening_and_refinement();
for(unsigned int fieldIndex=0; fieldIndex<fields.size(); fieldIndex++){
	// The following lines were from an earlier version.
	// residualSet is cleared in reinit(), so I don't see the reason for the pointer assignment
	// Changing to the new version has no impact on the solution.
	//(*residualSet[fieldIndex])=(*solutionSet[fieldIndex]);
	//soltransSet[fieldIndex]->prepare_for_coarsening_and_refinement(*residualSet[fieldIndex]);

	soltransSet[fieldIndex]->prepare_for_coarsening_and_refinement(*solutionSet[fieldIndex]);
}
triangulation.execute_coarsening_and_refinement();

}

#include "../../include/matrixFreePDE_template_instantiations.h"
