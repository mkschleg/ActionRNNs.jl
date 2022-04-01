"""
Dealing with constructing all our components. How do we ensure
1.) Discoverability
2.) Sanity.

Detached from the actual structs so they can be used independently.
"""


# Dealing with construction on a massive scale...
include("construct/rnns.jl")
include("construct/learning_update.jl")

