# 📊 Plot Overview — LSM Project 1 Experiments
---

---

## 2.1 Effects of chunk size 

### **2.2 Computation vs. Communication Breakdown**
- **Goal:** Quantify the ratio of computation to communication time.
- **Type:** `sns.barplot` (grouped or stacked)
- **Axes:** x = Rank, y = Time (s)
- **Hue:** Metric (`Comp Total`, `Comm Total`)
- **Annotation** Annotate with chunck count pr rank

legend for config 
relplot for forskellige chunk sizes (subplots)

### **2.4 Heatmap of Chunk Times**
- **Goal:** Visualize spatial imbalance in chunk compute times.
- **Type:** `sns.heatmap`
- **Axes:** Chunk size / Image size × Chunk ID grid (keep chunk size OR Image Size constant)
- **Value:** Chunk compute time (s)

Relplot for domain (subplot) 

---
---

## 4. Scaling Studies

### **4.1 Problem Size Scaling**
- **Goal:** Assess strong scaling w.r.t. total image size.
- **Type:** `sns.lineplot` (log–log)
- **Axes:** x = Pixels (width × height), y = Wall Time (s)
- **Hue:** Config 
- **Style:** Image Size
- **Expected trend:** Linear scaling up to a point; communication dominates for very large images.
--- Basicly confirm den scaling vi så fra baseline vs numba 


---

## 5. Parallel Scaling (Ranks Study)

### **5.1 Runtime vs. Number of Ranks**
- **Goal:** Analyze parallel scaling efficiency.
- **Type:** `sns.lineplot` (log-log)
- **Axes:** x = N Ranks, y = Wall Time (s)
- **Hue:** config
- **Style:** Image size 

### **5.2 Communication Fraction vs. Ranks**
- **Goal:** Quantify how communication dominates at higher rank counts.
- **Type:** `sns.barplot`
- **Axes:** x = N Ranks, y = time with comm/comp breakdown  
- **Hue:** Config 

- **Expected trend:** Communication fraction grows with rank count.

---

## 6. Supplementary & Debug Plots

### **6.1 Rank-Level Gantt Overview**
- **Goal:** Visual overview of rank activity timeline (optional if time permits).
- **Type:** `matplotlib.broken_barh`
- **Axes:** Rank (y) × Time (x)
- **Value:** Task segments per chunk
- **Expected trend:** Dynamic scheduling fills idle gaps more effectively.

---

---

