# Finalized Scripts
## SCRIPT DESCRIPTIONS HERE ##
```func_1D_LiS.py``` & ```func_potentials.py```
  - ```func_potentials.py``` contains the equations and definitions for the initial conditions for the solid-state and electrolyte potentials, while ```func_1D_LiS.py``` contains the overall equations and complete discretization as well as residual vector and Jacobian matrix formulations.

```LiS_Model.py``` & ```LiS_Potential_Model.py```
  - ```LiS_Potential_Model.py``` contains the model class for the initial potentials problem, while ```LiS_Model.py``` is the model class for the overall 1D Li-S model which contains the substitution algorithm for the residual vector and Jacobian matrix.

```solver_LiS_1D_test.py``` & ```solver_potentials_test.py```
  - ```solver_potentials_test.py``` contains the solver function for the initial potentials problem, while ```solver_LiS_1D_test.py``` contains the finalized solver for the overall 1D Li-S model.

 ```generate_square_wave.py```
   - Contains functions for non-constant current formulations.

```further_solver.py```
  - Contains the solver function to be implemented for further simulations from the endpoint of another simulation result.

The remaining scripts are used for data analysis of the results obtained which are saved into ```.npz``` files.

### Example Result Files ###
```var_array_0.5A_constant.npz```
  - Result for 0.5A constant current discharge.

```var_array_0.5A_no_prep.npz```
  - Result for 0.5A constant current discharge without precipitation/dissolution dynamics.

```var_array_0.5A_GITT.npz```
  - Result for 0.5A-0A GITT current profile.

```var_array_0.25A_sine.npz```
  - Result for 0.25A sinusoidal current profile.
