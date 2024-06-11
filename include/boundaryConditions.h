#ifndef INCLUDE_BOUNDARYCONDITIONS_H_
#define INCLUDE_BOUNDARYCONDITIONS_H_

#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with the boundary conditions. Add more comments later
 */
template <int dim, int degree>
class boundaryConditions {
public:
    boundaryConditions(const userInputParameters<dim>& _userInputs);


private:
    /*User inputs*/
    userInputParameters<dim> userInputs;

};

template <int dim, int degree>
boundaryConditions<dim, degree>::boundaryConditions(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
{
}

#endif
