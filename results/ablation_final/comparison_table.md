# Final Ablation Study Summary (Dark Condition)
*All models tuned via Grid Search for best performance.*

|    | Model             | Dry_Weight R2   |   Dry_Weight RMSE | Oxygen_Rate R2   |   Oxygen_Rate RMSE | Chl_Per_Cell R2   |   Chl_Per_Cell RMSE | Fv_Fm R2       |   Fv_Fm RMSE |
|----|-------------------|-----------------|-------------------|------------------|--------------------|-------------------|---------------------|----------------|--------------|
|  0 | Step 1: Static *  | 0.874 ± 0.025   |             0.253 | 0.949 ± 0.007    |              0.111 | 0.905 ± 0.016     |               0.008 | 0.634 ± 0.069  |        0.002 |
|  1 | Step 2: GRU *     | 0.956 ± 0.029   |             0.144 | 0.981 ± 0.002    |              0.068 | 0.950 ± 0.034     |               0.006 | -1.392 ± 2.002 |        0.005 |
|  2 | Step 3: ODE-RNN * | 0.998 ± 0.000   |             0.03  | 0.981 ± 0.000    |              0.068 | 0.984 ± 0.000     |               0.003 | 0.835 ± 0.002  |        0.001 |