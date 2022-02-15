# General Documentation

This page hosts the general documentation of the `ActionRNNs.jl` library. This includes all research code used in this project.

## Contents

```@contents
Pages = ["library.md"]
```

## Index

```@index
Pages = ["library.md"]
```

## Cells

```@docs
ActionRNNs.AbstractActionRNN
ActionRNNs._needs_action_input
```

### Basic Cells
```@docs
ActionRNNs.AARNN
ActionRNNs.AAGRU
ActionRNNs.AALSTM
ActionRNNs.MARNN
ActionRNNs.MAGRU
ActionRNNs.MALSTM
ActionRNNs.FacMARNN
ActionRNNs.FacMAGRU
ActionRNNs.FacTucMARNN
ActionRNNs.FacTucMAGRU
```

### Combo Cells
```@docs
ActionRNNs.CaddRNN
ActionRNNs.CaddGRU
ActionRNNs.CaddAAGRU
ActionRNNs.CaddMAGRU
ActionRNNs.CaddElRNN
ActionRNNs.CaddElGRU
ActionRNNs.CcatRNN
ActionRNNs.CcatGRU
```

### Mixed Cells

```@docs
ActionRNNs.MixRNN
ActionRNNs.MixElRNN
ActionRNNs.MixGRU
ActionRNNs.MixElGRU
```

### Old/DefunctCells

```@docs
ActionRNNs.GAUGRU
ActionRNNs.GAIGRU
ActionRNNs.GAIARNN
ActionRNNs.GAIAGRU
ActionRNNs.GAIALSTM
```

### Shared operations for cells

```@docs
ActionRNNs.contract_WA
ActionRNNs.get_waa
```

## Other Layers

```@docs
ActionRNNs.ActionDense
```

## Learning Updates

```@docs
ActionRNNs.QLearning
```


## Constructors

```@docs
ActionRNNs.build_rnn_layer
```

## Agents

### Experience Replay Agents
```@docs
ActionRNNs.AbstractERAgent
```

#### Instantiations
```@docs
ActionRNNs.DRQNAgent
ActionRNNs.DRTDNAgent
```

#### Implementation details

```@docs
ActionRNNs.get_replay
ActionRNNs.get_learning_update
ActionRNNs.get_device
ActionRNNs.get_action_and_prob
ActionRNNs.get_model
ActionRNNs.MinimalRLCore.start!(agent::ActionRNNs.AbstractERAgent, s, rng; kwargs...)
ActionRNNs.MinimalRLCore.step!(agent::ActionRNNs.AbstractERAgent, env_s_tp1, r, terminal, rng; kwargs...)
ActionRNNs.MinimalRLCore.step!
ActionRNNs.update!(agent::ActionRNNs.AbstractERAgent{<:ActionRNNs.ControlUpdate}, rng)
ActionRNNs.update!(agent::ActionRNNs.AbstractERAgent{<:ActionRNNs.PredictionUpdate}, rng)
ActionRNNs.update!
ActionRNNs.update_target_network!
```


### Online Agents


### Tools/Utils

```@docs
ActionRNNs.UpdateTimer
ActionRNNs.make_obs_list
ActionRNNs.build_new_feat
```

##### Hidden state manipulation

```@docs
ActionRNNs.HSStale
ActionRNNs.HSMinimize
ActionRNNs.HSRefil
ActionRNNs.get_hs_replay_strategy
ActionRNNs.modify_hs_in_er!
ActionRNNs.modify_hs_in_er_by_grad!
ActionRNNs.reset!
```

##### Replay buffer

```@docs
ActionRNNs.CircularBuffer
ActionRNNs.StateBuffer
Base.length(buffer::ActionRNNs.CircularBuffer)
Base.push!(buffer::CB, data::NamedTuple) where {CB<:ActionRNNs.CircularBuffer}
ActionRNNs.get_hs_details_for_er
ActionRNNs.hs_symbol_layer
ActionRNNs.get_hs_symbol_list
ActionRNNs.get_state_from_experience
ActionRNNs.get_information_from_experience
ActionRNNs.make_replay
ActionRNNs.get_hs_from_experience!
ActionRNNs.capacity

```

##### Flux Chain Manipulation

```@docs
ActionRNNs.contains_comp
ActionRNNs.find_layers_with_eq
ActionRNNs.find_layers_with_recur
ActionRNNs.contains_rnn_type
ActionRNNs.needs_action_input
ActionRNNs.contains_layer_type
```

### Policies

```@docs
ActionRNNs.ϵGreedy
ActionRNNs.ϵGreedyDecay
ActionRNNs.get_prob
ActionRNNs.sample
```

### Feature Constructors


## Environments

### TMaze

```@docs
ActionRNNs.TMaze
```

### DirectionalTMaze

```@docs
ActionRNNs.DirectionalTMaze
```

### Masked Grid World

```@docs
ActionRNNs.MaskedGridWorld
ActionRNNs.MaskedGridWorldHelpers
```

### Lunar Lander

```@docs
ActionRNNs.LunarLander
```

## FluxUtils Stuff

```@docs
ActionRNNs.ExpUtils.FluxUtils.get_optimizer
ActionRNNs.ExpUtils.FluxUtils.RMSPropTF
ActionRNNs.ExpUtils.FluxUtils.RMSPropTFCentered
```

## Misc

