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


### Shared operations for cells

```@docs
ActionRNNs.contract_WA
ActionRNNs.get_waa
```

## Constructors

```@docs
ActionRNNs.build_rnn_layer
```

## Agents

```@docs
ActionRNNs.MinimalRLCore.AbstractAgent
```

### Experience Replay Agents
```@docs
ActionRNNs.AbstractERAgent
```

```@docs
ActionRNNs.get_replay_buffer
ActionRNNs.get_learning_update
ActionRNNs.get_device
ActionRNNs.get_action_and_prob
ActionRNNs.MinimalRLCore.start!(agent::ActionRNNs.AbstractERAgent, s, rng; kwargs...)
ActionRNNs.MinimalRLCore.step!(agent::ActionRNNs.AbstractERAgent, env_s_tp1, r, terminal, rng; kwargs...)
ActionRNNs.update!(agent::ActionRNNs.AbstractERAgent{<:ActionRNNs.ControlUpdate}, rng)
ActionRNNs.update!(agent::ActionRNNs.AbstractERAgent{<:ActionRNNs.PredictionUpdate}, rng)
ActionRNNs.update_target_network!
```

#### Instantiations
```@docs
ActionRNNs.DRQNAgent
ActionRNNs.DRTDNAgent
```

### Online Agents


### Tools/Utils

```@docs
ActionRNNs.UpdateTimer
ActionRNNs.make_obs_list
ActionRNNs.obs_init
ActionRNNs.image_init
ActionRNNs.make_replay
ActionRNNs.get_state_from_experience
ActionRNNs.get_information_from_experience
ActionRNNs.build_new_feat
```

### Feature Constructors

<!-- ```@docs -->
<!-- ActionRNNs.IdentityFeatureCreator -->
<!-- ActionRNNs.AddDimFeatureCreator -->
<!-- ``` -->

## Environments

## Misc Stuff

# Misc Tools





