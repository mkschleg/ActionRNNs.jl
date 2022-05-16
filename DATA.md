
# RingWorld

- Standard: `"cell" => ["AAGRU" ,"AARNN" ,"GRU" ,"MAGRU" ,"MARNN" ,"RNN"]`
- Factored: 

- BaseDirectory: `ActionRNNs.jl/local_data/ringworld`

## DATA

**Experience Replay:**

| Description                             | Computer  | Path                                       | Database | Analyzing File File            |
|:----------------------------------------|-----------|:-------------------------------------------|:---------|--------------------------------|
| Sensitivity (FacRNN) Same size as AARNN | GPUServer | `ringworld_fac_er_rmsprop_10.tar.gz`       | NA       | `ringworld/ringworld_plots.jl` |
| Sensitivity (FacRNN) Same size as MARNN | GPUServer | `ringworld_fac_er_rmsprop_10_marnn.tar.gz` | NA       | "                              |
| Sensitivity (FacGRU) Same size as MAGRU | GPUServer | `ringworld_fac_er_rmsprop_10_magru.tar.gz` | NA       | "                              |
| Sensitivity (FacGRU) Same size as AAGRU | GPUServer | `ringworld_fac_er_rmsprop_10_aagru.tar.gz` | NA       | "                              |
| Sensitivity (Factored) Init style       | GPUserver | `ringworld_er_rmsprop_10_fac.tar.gz`       | NA       | `ringworld/ringworld_fac.jl`   |
| Final (Standard)                        | GPUServer | `final_ringworld_er_rmsprop_10.tar.gz`     | NA       | `ringworld/ringworld_plots.jl` |


**Online:**

| Description                       | Computer  | Path                                     | Database | Analyzing File File          |
|:----------------------------------|-----------|:-----------------------------------------|:---------|------------------------------|
| Sensitivity (Factored) Init style | GPUserver | `ringworld_online_rmsprop_10_fac.tar.gz` | NA       | `ringworld/ringworld_fac.jl` |
|                                   |           |                                          |          |                              |

## Notebooks
- `ringworld_heatmaps.jl`
- `ringworld_plots.jl`
- `ringworld_sensitivity_plots.jl`
- `ringworld_fac.jl`
- `ringworld_matt.jl`
- `ringworld_ridgeline_plots.jl`

# TMaze

# Directional TMaze

- BaseDirectory: `ActionRNNs.jl/local_data/dir_tmaze_<er/online>`

## Data

**Experience Replay:**

| Description                             | Computer  | Path                                                  | Database | Analyzing File File  |
|:----------------------------------------|:----------|:------------------------------------------------------|:---------|----------------------|
| Sweeps Factored w/ Adam                 | GPUServer | `dir_tmaze_er_fac_rnn_adam_10.tar.gz`                 | NA       | `dir_tmaze_plots.jl` |
| Sweeps Factored Init w/ RMSprop         | GPUServer | `dir_tmaze_er_fac_rnn_init_rmsprop_10.tar.gz`         | NA       | `dir_tmaze_plots.jl` |
| Sweeps Factored w/ RMSProp              | GPUServer | `dir_tmaze_er_fac_rnn_rmsprop_10(_p2).tar.gz`         | NA       | `dir_tmaze_plots.jl` |
| Sweeps Usual Cells w/ buffer = 20k      | GPUServer | `dir_tmaze_er_rnn_rmsprop_10_20k.tar.gz`              | NA       | `dir_tmaze_plots.jl` |
| Sweeps Usual Cells w/ buffer_size = 50k | GPUServer | `dir_tmaze_er_rnn_rmsprop_10_50k.tar.gz`              | NA       | `dir_tmaze_plots.jl` |
| Sweeps Usual Cells w/ buffer_size=10k   | GPUServer | `dir_tmaze_er_rnn_rmsprop_10(_p2).tar.gz`             | NA       | `dir_tmaze_plots.jl` |
| Sweeps Usual Cells w/ size=6            | GPUServer | `dir_tmaze_er_rnn_rmsprop.tar.gz`                     | NA       | `dir_tmaze_plots.jl` |
| Final size=6                            | GPUServer | `final_dir_tmaze_6_er_rnn_rmsprop.tar.gz`             | NA       | `final_dir_plots.jl` |
| Final buffer_size=50k                   | GPUServer | `final_dir_tmaze_er_rnn_rmsprop_10_2_50k.tar.gz`      | NA       | `final_dir_plots.jl` |
| Final buffer_size=10k                   | GPUServer | `final_dir_tmaze_er_rnn_rmsprop_10_2.tar.gz`          | NA       | `final_dir_plots.jl` |
| Final buffer_size=10k                   | GPUServer | `final_dir_tmaze_er_rnn_rmsprop_10.tar.gz`            | NA       | `final_dir_plots.jl` |
| Final factored                          | GPUServer | `final_fac_dir_tmaze_er_rnn_rmsprop_10_2_300k.tar.gz` | NA       | `final_dir_plots.jl` |
| Final factored                          | GPUServer | `final_fac_dir_tmaze_er_rnn_rmsprop_10_2.tar.gz`      | NA       | `final_dir_plots.jl` |

**Online:**

| Description | Computer  | Path                                               | Database | Analyzing File File         |
|:------------|:----------|:---------------------------------------------------|:---------|-----------------------------|
|             | GPUServer | `dir_tmaze_online_rmsprop_10_facmagru_t16.tar.gz`  | NA       | `dir_tmaze_online_plots.jl` |
|             | GPUServer | `dir_tmaze_online_rmsprop_10_facmarnn_t16.tar.gz`  | NA       | `dir_tmaze_online_plots.jl` |
|             | GPUServer | `dir_tmaze_online_rmsprop_10_fac.tar.gz`           | NA       | `dir_tmaze_online_plots.jl` |
|             | GPUServer | `dir_tmaze_online_rmsprop_size10_500k_nh15.tar.gz` | NA       | `dir_tmaze_online_plots.jl` |
|             | GPUServer | `final_dir_tmaze_online_rmsprop_10_t16.tar.gz`     | NA       | `dir_tmaze_online_plots.jl` |
|             | GPUServer | `final_dir_tmaze_online_rmsprop_10.tar.gz`         | NA       | `dir_tmaze_online_plots.jl` |


# Masked Grid World

| Description | Computer | Path | Database | Analyzing File File |
|:------------|:---------|:-----|:---------|---------------------|
|             |          |      |          |                     |

# Image DirTMaze

| Description                                            | Computer  | Path                                           | Database | Analyzing File       |
|:-------------------------------------------------------|:----------|:-----------------------------------------------|:---------|----------------------|
| SWEEP: Standard cells for image dirtmaze with size = 6 | gpuserver | `image_tmaze/image_dir_tmaze_adam_6`           | NA       | `image_dir_tmaze.jl` |
| SWEEP: Factored cells and initializations for ""       | gpuserver | `image_tmaze/image_dir_tmaze_adam_6_init`      | NA       | `image_dir_tmaze.jl` |
| FINAL: Standard cells                                  | gpuserver | `image_tmaze/final_image_dir_tmaze_adam_6`     | NA       | `image_dir_tmaze.jl` |
| FINAL: Factored cells                                  | gpuserver | `image_tmaze/final_image_fac_dir_tmaze_adam_6` | NA       | `image_dir_tmaze.jl` |





# Lunar Lander

