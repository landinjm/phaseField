// methods to apply boundary conditons

#include "../../include/matrixFreePDE.h"
#include "../../include/nonUniformDirichletBC.h"
#include "../../include/varBCs.h"
#include "../../include/vectorBCFunction.h"

// =================================================================================
// Methods to apply non-zero Dirichlet BCs
// =================================================================================
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::applyDirichletBCs()
{
    // First, get the variable index of the current field
    unsigned int starting_BC_list_index = 0;

    for (unsigned int i = 0; i < currentFieldIndex; i++) {

        if (userInputs.var_type[i] == SCALAR) {
            starting_BC_list_index++;
        } else {
            starting_BC_list_index += dim;
        }
    }

    if (userInputs.var_type[currentFieldIndex] == SCALAR) {
        for (unsigned int direction = 0; direction < 2 * dim; direction++) {
            if (userInputs.BC_list[starting_BC_list_index].var_BC_type[direction] == DIRICHLET) {
                VectorTools::interpolate_boundary_values(*Discretization.dofHandlersSet[currentFieldIndex],
                    direction, Functions::ConstantFunction<dim>(userInputs.BC_list[starting_BC_list_index].var_BC_val[direction], 1), *(AffineConstraints<double>*)BCs.constraintsDirichletSet[currentFieldIndex]);

            } else if (userInputs.BC_list[starting_BC_list_index].var_BC_type[direction] == NON_UNIFORM_DIRICHLET) {
                VectorTools::interpolate_boundary_values(*Discretization.dofHandlersSet[currentFieldIndex],
                    direction, NonUniformDirichletBC<dim, degree>(currentFieldIndex, direction, tStep.currentTime, this), *(AffineConstraints<double>*)BCs.constraintsDirichletSet[currentFieldIndex]);
            }
        }
    } else {
        for (unsigned int direction = 0; direction < 2 * dim; direction++) {

            std::vector<double> BC_values;
            for (unsigned int component = 0; component < dim; component++) {
                BC_values.push_back(userInputs.BC_list[starting_BC_list_index + component].var_BC_val[direction]);
            }

            std::vector<bool> mask;
            for (unsigned int component = 0; component < dim; component++) {
                if (userInputs.BC_list[starting_BC_list_index + component].var_BC_type[direction] == DIRICHLET) {
                    mask.push_back(true);
                } else {
                    mask.push_back(false);
                }
            }

            VectorTools::interpolate_boundary_values(*Discretization.dofHandlersSet[currentFieldIndex],
                direction, vectorBCFunction<dim>(BC_values), *(AffineConstraints<double>*)BCs.constraintsDirichletSet[currentFieldIndex], mask);

            // Mask again, this time for non-uniform Dirichlet BCs
            mask.clear();
            for (unsigned int component = 0; component < dim; component++) {
                if (userInputs.BC_list[starting_BC_list_index + component].var_BC_type[direction] == NON_UNIFORM_DIRICHLET) {
                    mask.push_back(true);
                } else {
                    mask.push_back(false);
                }
            }

            // VectorTools::interpolate_boundary_values (*Discretization.dofHandlersSet[currentFieldIndex],\
				//   direction, NonUniformDirichletBC<dim,degree>(currentFieldIndex,direction,tStep.currentTime,this), *(AffineConstraints<double>*) \
				//   BCs.constraintsDirichletSet[currentFieldIndex],mask);
            VectorTools::interpolate_boundary_values(*Discretization.dofHandlersSet[currentFieldIndex],
                direction, NonUniformDirichletBCVector<dim, degree>(currentFieldIndex, direction, tStep.currentTime, this), *(AffineConstraints<double>*)BCs.constraintsDirichletSet[currentFieldIndex], mask);
        }
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
