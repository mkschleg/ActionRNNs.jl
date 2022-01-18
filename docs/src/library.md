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

# ActionRNNs.jl

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

## Environments

## Misc Stuff

# Misc Tools





