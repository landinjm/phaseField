#ifndef INCLUDE_ADAPTIVEREFINEMENT_H_
#define INCLUDE_ADAPTIVEREFINEMENT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "fields.h"
#include "userInputParameters.h"

#ifndef vectorType
typedef dealii::LinearAlgebra::distributed::Vector<double> vectorType;
#endif

using namespace dealii;

/**
 * This class deals with adaptive refinement. Add more comments later
 */
template <int dim, int degree>
class adaptiveRefinement {
public:
    adaptiveRefinement(const userInputParameters<dim>& _userInputs, parallel::distributed::Triangulation<dim>& _triangulation, std::vector<Field<dim>>& _fields, std::vector<vectorType*>& _solutionSet, std::vector<parallel::distributed::SolutionTransfer<dim, vectorType>*>& _soltransSet, std::vector<FESystem<dim>*>& _FESet, std::vector<DoFHandler<dim>*>& _dofHandlersSet_nonconst, std::vector<const AffineConstraints<double>*>& _constraintsDirichletSet);

    /*Adaptive refinement*/
    void adaptiveRefine(unsigned int _currentIncrement);

    /*Method that refines the triangulation*/
    void refineGrid();

    /*Current increment*/
    unsigned int currentIncrement;

    /*A vector of all the hanging node constraints for adaptive meshes in the problem. A constraint set is a map which holds the mapping between the degrees of freedom and the corresponding degree of freedom constraints.*/
    std::vector<const AffineConstraints<double>*> constraintsOtherSet;

    /*Copies of constraintSet elements, but stored as non-const to enable application of constraints.*/
    std::vector<AffineConstraints<double>*> constraintsOtherSet_nonconst;


private:
    // Adaptive refinement criterion
    void adaptiveRefineCriterion();

    userInputParameters<dim> userInputs;

    parallel::distributed::Triangulation<dim>& triangulation;

    std::vector<Field<dim>>& fields;

    std::vector<vectorType*>& solutionSet;

    std::vector<parallel::distributed::SolutionTransfer<dim, vectorType>*>& soltransSet;

    std::vector<FESystem<dim>*>& FESet;

    std::vector<DoFHandler<dim>*>& dofHandlersSet_nonconst;

    std::vector<const AffineConstraints<double>*>& constraintsDirichletSet;
};

template <int dim, int degree>
adaptiveRefinement<dim, degree>::adaptiveRefinement(const userInputParameters<dim>& _userInputs, parallel::distributed::Triangulation<dim>& _triangulation, std::vector<Field<dim>>& _fields, std::vector<vectorType*>& _solutionSet, std::vector<parallel::distributed::SolutionTransfer<dim, vectorType>*>& _soltransSet, std::vector<FESystem<dim>*>& _FESet, std::vector<DoFHandler<dim>*>& _dofHandlersSet_nonconst, std::vector<const AffineConstraints<double>*>& _constraintsDirichletSet)
    : userInputs(_userInputs)
    , triangulation(_triangulation)
    , fields(_fields)
    , solutionSet(_solutionSet)
    , soltransSet(_soltransSet)
    , FESet(_FESet)
    , dofHandlersSet_nonconst(_dofHandlersSet_nonconst)
    , constraintsDirichletSet(_constraintsDirichletSet)
{
}

template <int dim, int degree>
void adaptiveRefinement<dim, degree>::adaptiveRefine(unsigned int currentIncrement)
{
    if ((currentIncrement == 0)) {
        adaptiveRefineCriterion();
        refineGrid();
    } else {
        // Apply constraints before remeshing
        for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
            constraintsDirichletSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
            constraintsOtherSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
            solutionSet[fieldIndex]->update_ghost_values();
        }
        adaptiveRefineCriterion();
        refineGrid();
    }
}

template <int dim, int degree>
void adaptiveRefinement<dim, degree>::adaptiveRefineCriterion()
{
    QGaussLobatto<dim> quadrature(degree + 1);
    const unsigned int num_quad_points = quadrature.size();

    // Set the correct update flags & grab the indices of the scalar and vector fields if any
    bool need_value = false;
    bool need_gradient = false;
    for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); ++it) {
        if (it->criterion_type == VALUE || it->criterion_type == VALUE_AND_GRADIENT) {
            need_value = true;
        } else if (it->criterion_type == GRADIENT || it->criterion_type == VALUE_AND_GRADIENT) {
            need_gradient = true;
        }
    }
    dealii::UpdateFlags update_flags;
    if (need_value && !need_gradient) {
        update_flags = update_values;
    } else if (!need_value && need_gradient) {
        update_flags = update_gradients;
    } else {
        update_flags = update_values | update_gradients;
    }

    //Before marking cells for refinement and/or coarsening clear user flags
    //These user flags are used to mark whether cells have already been flagged for refinement
    triangulation.clear_user_flags();

    for (auto it = userInputs.refinement_criteria.begin(); it != userInputs.refinement_criteria.end(); ++it) {

        //Grab the field type
        unsigned int fieldType = userInputs.var_type[it->variable_index];

        //Grab the field index
        unsigned int index = it->variable_index;

        FEValues<dim> fe_values(*FESet[index], quadrature, update_flags);

        std::vector<double> values(num_quad_points);
        std::vector<double> gradient_magnitudes(num_quad_points);

        std::vector<dealii::Vector<double>> values_vector(num_quad_points, dealii::Vector<double>(dim));
        dealii::Vector<double> gradient_magnitude_components(dim);
        std::vector<dealii::Tensor<1, dim, double>> gradients(num_quad_points);
	    std::vector<std::vector<dealii::Tensor<1,dim,double>>> gradients_vector(num_quad_points, std::vector<dealii::Tensor<1,dim,double>>(dim));

        typename DoFHandler<dim>::active_cell_iterator cell = dofHandlersSet_nonconst[index]->begin_active(), endc = dofHandlersSet_nonconst[index]->end();

        typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell = triangulation.begin_active();

        //Loop through locally owned cells
        for (; cell != endc; ++cell) {
            if (cell->is_locally_owned()) {
                fe_values.reinit(cell);

                if (need_value && fieldType == SCALAR) {
                    fe_values.get_function_values(*solutionSet[it->variable_index], values);

                } else if (need_value && fieldType == VECTOR) {
                    fe_values.get_function_values(*solutionSet[it->variable_index], values_vector);

                    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
                        values.at(q_point) = values_vector.at(q_point).l2_norm();
                    }
                }
                if (need_gradient && fieldType == SCALAR) {
                    fe_values.get_function_gradients(*solutionSet[it->variable_index], gradients);

                    for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point) {
                        gradient_magnitudes.at(q_point) = gradients.at(q_point).norm();
                    }
                } else if (need_gradient && fieldType == VECTOR) {
                    fe_values.get_function_gradients(*solutionSet[it->variable_index], gradients_vector);

                    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
                        for (unsigned int d = 0; d<dim; ++d){
                            gradient_magnitude_components[d] = gradients_vector.at(q_point).at(d).norm();
                        }
                        gradient_magnitudes.at(q_point) = gradient_magnitude_components.l2_norm();
                    }
                }

                bool mark_refine = false;

                for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point) {
                    if (it->criterion_type == VALUE || it->criterion_type == VALUE_AND_GRADIENT) {
                        if ((values[q_point] > it->value_lower_bound) && (values[q_point] < it->value_upper_bound)) {
                            mark_refine = true;
                            break;
                        }
                    }
                    if (it->criterion_type == GRADIENT || it->criterion_type == VALUE_AND_GRADIENT) {
                        if (gradient_magnitudes[q_point] > it->gradient_lower_bound) {
                            mark_refine = true;
                            break;
                        }
                    }
                }

                // limit the maximal and minimal refinement depth of the mesh
                unsigned int current_level = t_cell->level();

                if (mark_refine && current_level < userInputs.max_refinement_level) {
                    cell->set_user_flag();
                    cell->clear_coarsen_flag();
                    cell->set_refine_flag();
                } else if (mark_refine) {
                    cell->set_user_flag();
                    cell->clear_coarsen_flag();
                } else if (!mark_refine && current_level > userInputs.min_refinement_level && !cell->user_flag_set()) {
                    cell->set_coarsen_flag();
                }
            }
            ++t_cell;
        }
    }
}

template <int dim, int degree>
void adaptiveRefinement<dim, degree>::refineGrid()
{
    // prepare and refine
    triangulation.prepare_coarsening_and_refinement();
    for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {

        soltransSet[fieldIndex]->prepare_for_coarsening_and_refinement(*solutionSet[fieldIndex]);
    }
    triangulation.execute_coarsening_and_refinement();
}

#endif
