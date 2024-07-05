#ifndef INCLUDE_TIMESTEPPING_H_
#define INCLUDE_TIMESTEPPING_H_

#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with the timestepping. Add more comments later
 */
template <int dim, int degree>
class TimeStepping {
public:
    TimeStepping(const userInputParameters<dim>& _userInputs);

    /*Vector all the solution vectors in the problem. In a multi-field problem, each primal field has a solution vector associated with it.*/
    std::vector<vectorType*> solutionSet;

    double currentTime;

    unsigned int currentIncrement;

private:
    /*User inputs*/
    userInputParameters<dim> userInputs;
};

template <int dim, int degree>
TimeStepping<dim, degree>::TimeStepping(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
    , currentTime(0.0)
    , currentIncrement(0)
{
}

#endif
