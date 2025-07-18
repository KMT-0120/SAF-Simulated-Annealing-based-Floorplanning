# SAF: A 3-Stage Simulated Annealing-based Floorplanner

This document provides a detailed description of the 3-stage floorplanning process implemented in `floorplanning(penalty).py`. The process leverages a multi-stage, multiprocessing-based Simulated Annealing (SA) algorithm to progressively refine a chip layout. Its core strength lies in the **K-Parent Based Parallel Search** strategy and a dual-layered **Pruning (가지치기)** mechanism to enhance search efficiency and avoid local optima.

---

### **🚀 Core Strategy 1: K-Parent Based Parallel Search**

This floorplanner moves beyond a single-solution search by employing a parallel processing strategy. Instead of evolving one layout, it maintains multiple independent candidate solutions, called **Parents**.

* **What is a Parent?**: Each `Parent` is a complete and independent `Chip` state, with its own unique B\*-Tree layout and associated cost.
* **How it Works**:
    1.  **Initialization**: The algorithm creates `k` initial parent states (where `k` is typically set based on the number of available CPU cores).
    2.  **Parallel Exploration**: At each iteration of the SA, the `k` parents are distributed among a pool of worker processes.
    3.  **Worker Task**: Each worker process takes one parent state and runs an independent, short-term SA search to try and improve it.
    4.  **Aggregation**: After the workers complete, the best layout found by each is returned, forming the next generation of `k` parents.
    5.  **Global Best**: The single best solution found across all workers is tracked as the `global_best_chip`.
* **Advantage**: This parallel approach allows the algorithm to explore different regions of the vast solution space simultaneously, making the search more robust and efficient.

---

### **✂️ Core Strategy 2: Dual Pruning Mechanism**

To optimize the search, the algorithm employs two distinct types of pruning:

#### **Internal Pruning (Worker-Level)**

* **Where it Happens**: Occurs *inside* each individual worker process (`worker_sa_depth_search`).
* **What it Does**: When a worker begins its search, it saves its initial starting state. After completing its assigned search depth, it checks if it found any solution that is *strictly better* (i.e., lower cost) than its starting state. If no direct improvement was made, the worker **discards all of its exploration** and reports back its original, unchanged starting state.
* **Purpose**: This is a quality-control mechanism. It prevents a search branch that only found worse or probabilistically accepted solutions from polluting the main parent pool. It prunes away futile exploration paths at the source.

#### **External Pruning (Parent-Pool-Level)**

* **Where it Happens**: Occurs in the *main controlling process* (`multiprocess_k_parent_sa`), operating on the entire pool of `k` parents.
* **What it Does**: Periodically (e.g., every 1000 iterations), the algorithm ranks all `k` parents by their cost. It then **discards the worst-performing parents**. To fill these empty slots, it takes the top-performing parents, creates copies, and applies random modifications (**Mutation**) to them.
* **Purpose**: This applies evolutionary pressure to the entire population of solutions. It eliminates unpromising evolutionary lines (bad parents) and reinforces successful ones by creating new variations of them. This guides the entire search population towards more promising regions of the solution space.

---

### **1. Initialization Stage**

This stage sets up the initial layouts for the parallel search.

1.  **File Parsing**: Parses `.yal` or GSRC files into `Module` objects.
2.  **Initial B\*-Tree Construction**: Creates `k` distinct, randomized B\*-Tree layouts to serve as the initial parents.
3.  **Dynamic Cost Scaling**: The `CostScaler` normalizes cost components (Area, HPWL, Penalty) for balanced evaluation.
4.  **(Optional) Partial SA**: A brief, parallel SA run improves the initial `k` parents, providing a stronger starting point.

---

### **2. Stage 1 SA: Broad Exploration** 🗺️

This stage uses the K-Parent strategy to broadly explore the solution space.

* **Parallel Strategy**: `k` parents evolve in parallel. High initial temperature encourages each worker to explore diverse configurations.
* **Cost Function**: Focuses on **Area** and **HPWL**, with low weight on **Dead Space**.
* **Pruning**: Primarily relies on **Internal Pruning** within workers to maintain the quality of exploration.

---

### **3. Stage 2 SA: Focused Search & Pruning** 🎯

This stage refines the diverse set of solutions from Stage 1.

* **Parallel Strategy**: The parallel search continues with more focus.
* **Pruning**: Both **Internal Pruning** (in workers) and **External Pruning** (on the parent pool) are actively used. External Pruning ensures that the collective search effort is concentrated on the most promising solution families.
* **Cost Function**: The **penalty weight is increased significantly** to enforce boundary constraints.

---

### **4. Stage 3 SA: Hybrid Fine-Tuning** 🔬

This final stage uses a modified parallel strategy for maximum precision.

* **Hybrid Parallel Strategy**:
    1.  **Parallel Exploitation**: All workers start from the single `global_best_chip` to intensively search its local neighborhood. **Internal Pruning** is critical here to ensure only true improvements are accepted.
    2.  **Serial Exploration**: A single "explorer" process searches independently, acting as a safeguard against the main pool getting stuck.
* **Cost Function**: Weights for both **Penalty** and **Dead Space** are at their maximum.

---

### **5. Final Compaction & Output** 📦

After all SA stages, the final `global_best_chip` is processed.

1.  **Greedy Compaction**: The `compact_floorplan_final` function pushes all modules left and then down to eliminate empty space.
2.  **Final Report**: Final metrics (Area, HPWL, Dead Space %) are reported, and the layout is saved.
