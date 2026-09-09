"""Virtual Compiler: executable, branchable surrogate of compilation.

Built on top of the AI-Compiler trusted kernel (``self_compiler``).
The real toolchain (gcc, plugin telemetry, verifier, policy lab,
differential gate) is the *oracle*. This package is the *virtual
environment*: cheap latent states, counterfactual branches, surrogate
predictions with uncertainty, budgeted search, and an agent swarm —
the ``observe -> learn -> represent -> branch -> intervene -> validate``
loop from the Virtual Surrogate program, instantiated for computation
(``Virtual Computer = intervene on a computational state``;
``Virtual Algorithm`` includes compilers).
"""

__version__ = "0.1.0"
