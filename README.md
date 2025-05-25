
SAF-Simulated-Annealing-based-Floorplanning


check directory example when you want to see B*-tree placement

use floorplanning(penalty).py when you want to check SAF

SAF consist of 3 process

1. make initial condition of chip using partial SA

2. 1st SA(iteration : 3000 ~ 5000 | include bound penalty, area, HPWL not include dead space) + compact

3. 2st SA(iteration : 30000~ 45000 | include bound penalty area, HPWL and dead space) + compact
