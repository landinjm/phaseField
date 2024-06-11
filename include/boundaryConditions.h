#ifndef INCLUDE_BOUNDARYCONDITIONS_H_
#define INCLUDE_BOUNDARYCONDITIONS_H_

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "userInputParameters.h"
#include "discretization.h"

using namespace dealii;

/**
 * This class deals with the boundary conditions. Add more comments later
 */
template <int dim, int degree>
class boundaryConditions {
public:
    boundaryConditions(const userInputParameters<dim>& _userInputs);

    /*A vector of all the constraint sets in the problem. A constraint set is a map which holds the mapping between the degrees of freedom and the corresponding degree of freedom constraints.*/
    std::vector<const AffineConstraints<double>*> constraintsDirichletSet;

    /*Copies of constraintSet elements, but stored as non-const to enable application of constraints.*/
    std::vector<AffineConstraints<double>*> constraintsDirichletSet_nonconst;

    /*Non-uniform boundary conditions function*/

    /*Method to apply boundary conditions*/

    /*Map of degrees of freedom to the corresponding Dirichlet boundary conditions, if any.*/

    /*Method for applying Dirichlet boundary conditions.*/

    /*Method for applying Neumann boundary conditions.*/
    void applyNeumannBCs(dealii::LinearAlgebra::distributed::Vector<double>&, unsigned int&);
    
private:
    /*User inputs*/
    userInputParameters<dim> userInputs;

    /*Discretiziation*/
    discretization<dim> Discretization;

};

template <int dim, int degree>
boundaryConditions<dim, degree>::boundaryConditions(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
    , Discretization(_userInputs)
{
}

template <int dim, int degree>
void boundaryConditions<dim, degree>::applyNeumannBCs(dealii::LinearAlgebra::distributed::Vector<double>& residual, unsigned int& currentFieldIndex)
{
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // NOTE: Currently this function doesn't work and it's call is commented out in solveIncrement.
    // The result is off by almost exactly a factor of 100,000. I don't know what the issue is.
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Check to the BC for the current field
    unsigned int starting_BC_list_index = 0;
    for (unsigned int i = 0; i < currentFieldIndex; i++) {
        if (userInputs.var_type[i] == SCALAR) {
            starting_BC_list_index++;
        } else {
            starting_BC_list_index += dim;
        }
    }

    if (userInputs.var_type[currentFieldIndex] != SCALAR) {
        std::cerr << "PRISMS-PF ERROR: Neumann boundary conditions are only supported for scalar fields." << std::endl;
        abort();
    }

    for (unsigned int direction = 0; direction < 2 * dim; direction++) {
        if (userInputs.BC_list[starting_BC_list_index].var_BC_type[direction] == NEUMANN) {

            FESystem<dim>* fe = Discretization.FESet[currentFieldIndex];
            QGaussLobatto<dim - 1> face_quadrature_formula(degree + 1);
            FEFaceValues<dim> fe_face_values(*fe, face_quadrature_formula, update_values | update_JxW_values);
            const unsigned int n_face_q_points = face_quadrature_formula.size(), dofs_per_cell = fe->dofs_per_cell;
            Vector<double> cell_rhs(dofs_per_cell);
            std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

            // Loop over each face on a boundary
            for (auto cell = Discretization.dofHandlersSet[0]->begin_active(); cell != Discretization.dofHandlersSet[0]->end(); ++cell) {
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
                    if (cell->face(f)->at_boundary()) {
                        if (cell->face(f)->boundary_id() == direction) {
                            fe_face_values.reinit(cell, f);
                            cell_rhs = 0.0;
                            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
                                double neumann_value = userInputs.BC_list[starting_BC_list_index].var_BC_val[direction];
                                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                    cell_rhs(i) += (neumann_value * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
                                }
                            }
                            cell->get_dof_indices(local_dof_indices);
                            // assemble
                            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                                residual[local_dof_indices[i]] += cell_rhs(i);
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif
