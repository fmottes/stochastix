"""ReactionNetwork class for simulating chemical reaction systems."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ControlTerm, MultiTerm, ODETerm, VirtualBrownianTree

from .._state_utils import pytree_to_state, state_to_pytree
from ..kinetics._massaction import (
    MassAction,
    mass_action_ode_rate,
    mass_action_propensity,
)
from ._reaction import Reaction

if TYPE_CHECKING:
    from .._stochsimsolve import SimulationResults


class ReactionNetwork(eqx.Module):
    """A network of chemical reactions for simulation.

    This class compiles a list of `Reaction` objects into a system ready for
    stochastic or deterministic simulation. It constructs the stoichiometric,
    reactant, and product matrices, and provides methods to compute propensities
    and simulate the system as an ordinary differential equation (ODE) or a
    stochastic differential equation (SDE).

    The network can be manipulated functionally, for example, by adding reactions
    or creating subnetworks through slicing.

    Note: State input formats.
        `vector_field` and `log_prob` accept any properly formatted pytree as state input, all other methods expect a flat array of species counts. You can use `pytree_to_state` and `state_to_pytree` to convert between the two formats.

    Attributes:
        reactions: A tuple of `Reaction` objects in the network.
        species: An ordered tuple of unique species names.
        volume: The system volume, used for concentration-dependent rates.
        stoichiometry_matrix: The matrix `S` where `S[i, j]` is the net change
            in species `i` from reaction `j`. Shape: `(n_species, n_reactions)`.
        reactant_matrix: The matrix of reactant stoichiometric coefficients.
            Shape: `(n_species, n_reactions)`.
        product_matrix: The matrix of product stoichiometric coefficients.
            Shape: `(n_species, n_reactions)`.
        n_reactions: The number of reactions in the network.
        n_species: The number of species in the network.

    Methods:
        propensity_fn: Computes the propensity vector for the network.
        vector_field: The vector field for deterministic ODE integration.
        ode_rhs: Alias for `vector_field`.
        drift_fn: The drift function for the Chemical Langevin Equation (CLE).
        cle_drift: Alias for `drift_fn`.
        noise_coupling: The noise coupling matrix for the Chemical Langevin Equation (CLE).
        cle_diffusion: Alias for `noise_coupling`.
        diffusion_fn: Alias for `noise_coupling`.
        diffusion_coeff_matrix: The diffusion coefficient matrix for the Fokker-Planck equation.
        diffusion_coeff: Alias for `diffusion_coeff_matrix`.
        diffrax_ode_term: Returns an `ODETerm` for ODE solving with `diffrax`.
        diffrax_sde_term: A `MultiTerm` for solving the system as an SDE with `diffrax`.
        log_prob: Calculates the log-probability of a simulation trajectory.
        copy: Creates a deep copy of the reaction network.
        to_latex: Generates a LaTeX representation of the reaction network.
    """

    reactions: tuple[Reaction, ...]
    species: tuple[str, ...] = eqx.field(static=True)
    volume: float
    stoichiometry_matrix: list[list[int | float]]  # species x reactions
    reactant_matrix: list[list[int | float]]  # species x reactions
    product_matrix: list[list[int | float]]  # species x reactions
    n_reactions: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    _species_map: dict = eqx.field(static=True)
    _named_reactions: dict = eqx.field(static=True)
    _all_mass_action: bool = eqx.field(static=True)

    def __init__(self, reactions: Iterable[Reaction], volume: float = 1.0):
        """Initializes the ReactionNetwork.

        Args:
            reactions: An iterable of `Reaction` objects.
            volume: The system volume. Affects concentration-dependent rates.
        """
        self.reactions = tuple(reactions)
        self.volume = volume

        # --- Upward Flow: Compile static information from reactions ---

        # 1. Discover all unique species from reactants, products, and kinetics requirements.
        all_species = set()
        for r in reactions:
            all_species.update(r.required_species)

        self.species = tuple(sorted(list(all_species)))
        self._species_map = {s: i for i, s in enumerate(self.species)}
        self.n_species = len(self.species)
        self.n_reactions = len(reactions)

        self._all_mass_action = all(
            isinstance(r.kinetics, MassAction) for r in reactions
        )

        # "Prepare" reactions by equipping kinetics with species indices
        prepared_reactions = []
        for r in reactions:
            prepared_kinetics = r.kinetics._bind_to_network(self._species_map)
            # Reaction is an equinox Module, so we use functional updates
            new_r = eqx.tree_at(
                lambda reaction: reaction.kinetics, r, prepared_kinetics
            )
            prepared_reactions.append(new_r)
        self.reactions = tuple(prepared_reactions)

        # 2. Build global structural matrices directly in (species x reactions) convention.
        reactant_matrix = jnp.zeros((self.n_species, self.n_reactions))
        product_matrix = jnp.zeros((self.n_species, self.n_reactions))

        for i, r in enumerate(self.reactions):
            for species, coeff in r.reactants_and_coeffs:
                species_idx = self._species_map[species]
                reactant_matrix = reactant_matrix.at[species_idx, i].set(coeff)
            for species, coeff in r.products_and_coeffs:
                species_idx = self._species_map[species]
                product_matrix = product_matrix.at[species_idx, i].set(coeff)

        self.reactant_matrix = reactant_matrix.tolist()
        self.product_matrix = product_matrix.tolist()
        # Calculate stoichiometry from the jnp arrays before converting to list
        self.stoichiometry_matrix = (product_matrix - reactant_matrix).tolist()

        # Create a lightweight dictionary of named substeps with indices
        self._named_reactions = {}
        for i, reaction in enumerate(reactions):
            name = reaction.name
            if name is None:
                name = f'r{i}'
            elif name in self._named_reactions:
                j = 1
                while f'{name}_{j}' in self._named_reactions:
                    j += 1
                name = f'{name}_{j}'
            self._named_reactions[name] = i

        self.n_reactions = len(self.reactions)
        self.n_species = len(self.species)

    # --- Stochastic probensities ---

    def propensity_fn(
        self,
        x: jnp.ndarray,
        t: float = None,
        reaction_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Computes the propensity vector for the network.

        This function calculates the propensity `a(x, t)` for each reaction. It is
        optimized for the common case where all propensities are calculated
        (`reaction_mask=None`). For the masked case, it uses `jax.lax.cond` to
        remain JIT-compatible with dynamic masks.

        Args:
            x: The current state of the system (species counts).
            t: The current time.
            reaction_mask: An optional boolean array of shape `(n_reactions,)`
                indicating which propensities to compute. If `None`, all are computed.

        Returns:
            Propensities for each reaction. Propensities for masked-out reactions will be zero.
        """
        reactant_matrix = jnp.asarray(self.reactant_matrix)
        propensities = jnp.zeros(self.n_reactions)

        # This Python-level `if` allows JAX to compile a specialized, faster
        # version of the function for the common case where `mask` is None,
        # avoiding any `lax.cond` overhead.
        if reaction_mask is None:
            # Optimized path for the common case: no masking.
            if self._all_mass_action:
                # Vectorized computation for all mass-action reactions.
                rate_constants = jnp.array(
                    [r.kinetics.transform(r.kinetics.k) for r in self.reactions]
                )
                propensities = eqx.filter_vmap(
                    mass_action_propensity, in_axes=(0, None, 1, None)
                )(rate_constants, x, reactant_matrix, self.volume)
            else:
                # Path for heterogeneous reactions.
                for i, reaction in enumerate(self.reactions):
                    a = reaction.kinetics.propensity_fn(
                        x, reactant_matrix[:, i], t, volume=self.volume
                    )
                    propensities = propensities.at[i].set(a)
        else:
            reaction_mask = reaction_mask.astype(bool)
            # Path for the case where a mask is provided (e.g. next reaction method).
            if self._all_mass_action:
                # Vectorized approach with an inner lax.cond to handle the mask.
                # This is JIT-compatible and avoids a Python loop.
                def _propensity_fn_masked(k, reactants, masked):
                    return jax.lax.cond(
                        masked,
                        lambda: mass_action_propensity(k, x, reactants, self.volume),
                        lambda: 0.0,
                    )

                rate_constants = jnp.array(
                    [r.kinetics.transform(r.kinetics.k) for r in self.reactions]
                )
                propensities = eqx.filter_vmap(
                    _propensity_fn_masked, in_axes=(0, 1, 0)
                )(rate_constants, reactant_matrix, reaction_mask)
            else:
                # Path for heterogeneous reactions with masking.
                for i, reaction in enumerate(self.reactions):
                    # Use lax.cond for JIT-compatible conditional execution.
                    propensities = propensities.at[i].set(
                        jax.lax.cond(
                            reaction_mask[i],
                            lambda: reaction.kinetics.propensity_fn(
                                x, reactant_matrix[:, i], t, volume=self.volume
                            ),
                            lambda: 0.0,
                        )
                    )

        return propensities

    # --- Mean field formulation (ODE) ---

    def vector_field(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The vector field for deterministic ODE integration.

        This method computes the right-hand side of the ODE system, `dx/dt = f(t, x)`,
        based on the deterministic rate laws of the reactions. The signature
        `f(t, x, args)` is compatible with `diffrax.diffeqsolve`.

        Args:
            t: The current time.
            x: The current state vector (species counts).
            args: Additional arguments (not used).

        Returns:
            The time derivative of the state, `dx/dt`, in molecules/time (counts/time).
        """
        reactant_matrix = jnp.asarray(self.reactant_matrix)
        stoichiometry_matrix = jnp.asarray(self.stoichiometry_matrix)

        x = pytree_to_state(x, self.species)

        # numerical stability
        x = x + jnp.finfo(jnp.result_type(float)).tiny

        if self._all_mass_action:
            # Vectorized computation for all mass-action reactions.
            rate_constants = jnp.array(
                [r.kinetics.transform(r.kinetics.k) for r in self.reactions]
            )
            rates = eqx.filter_vmap(mass_action_ode_rate, in_axes=(0, None, 1, None))(
                rate_constants, x, reactant_matrix, self.volume
            )
        else:
            # This loop is JAX-jittable for heterogeneous reactions.
            rates = jnp.zeros(len(self.reactions))
            for i, reaction in enumerate(self.reactions):
                rate = reaction.kinetics.ode_rate_fn(
                    x, reactant_matrix[:, i], t, volume=self.volume
                )
                rates = rates.at[i].set(rate)

        dxdt = stoichiometry_matrix @ rates

        # convert back to pytree
        return state_to_pytree(x, self.species, dxdt)

    def ode_rhs(self, t: jnp.floating, x: jnp.ndarray, args: Any = None) -> jnp.ndarray:
        """The right-hand side of the ODE system. Alias for `vector_field`."""
        return self.vector_field(t, x, args)

    # --- Chemical Langevin Equation (SDE) ---

    def drift_fn(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The drift function for the Chemical Langevin Equation (CLE).

        The CLE is given by: `dX_t = f(t, X_t) dt + g(t, X_t) dW_t`. This
        function computes the drift term `f(t, X_t) = S * a(X_t, t)`, where `S`
        is the stoichiometry matrix and `a` is the propensity vector.

        Note:
            This is different from `vector_field`. The CLE drift is based on
            stochastic propensities, while the ODE vector field is based on
            deterministic rates. Better to use the `cle_drift` alias for clarity.

        Args:
            t: The current time.
            x: The current state vector (species counts).
            args: Additional arguments (not used).

        Returns:
            The drift vector for the CLE.
        """
        stoichiometry_matrix = jnp.asarray(self.stoichiometry_matrix)
        propensities = self.propensity_fn(x, t)

        return stoichiometry_matrix @ propensities

    def cle_drift(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The drift function `f(t, X_t)` for the Chemical Langevin Equation (CLE). Alias for `drift_fn`."""
        return self.drift_fn(t, x, args)

    def noise_coupling(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The noise coupling matrix for the Chemical Langevin Equation (CLE).

        The CLE is given by: `dX_t = f(t, X_t) dt + g(t, X_t) dW_t`. This
        function computes the noise coupling term `g(t, X_t)`, which is defined
        as `S * diag(sqrt(a(X_t, t)))`, where `S` is the stoichiometry matrix and
        `a` is the propensity vector.

        Note:
            This is **not** the same as the diffusion coefficient matrix used in
            the Fokker-Planck equation. See `diffusion_coeff_matrix`.

        Args:
            t: The current time.
            x: The current state vector (species counts).
            args: Additional arguments (not used).

        Returns:
            The noise coupling matrix of shape `(n_species, n_reactions)`.
        """
        stoichiometry_matrix = jnp.asarray(self.stoichiometry_matrix)
        propensities = self.propensity_fn(x, t)

        # Noise coupling matrix: S * diag(sqrt(propensities))
        # Shape: (n_species, n_reactions)
        sqrt_propensities = jnp.sqrt(jnp.maximum(propensities, 0.0))
        diffusion_matrix = stoichiometry_matrix * sqrt_propensities[None, :]

        return diffusion_matrix

    def cle_diffusion(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The diffusion function `g(t, X_t)` for the Chemical Langevin Equation (CLE). Alias for `noise_coupling`."""
        return self.noise_coupling(t, x, args)

    def diffusion_fn(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The diffusion function `g(t, X_t)` for the Chemical Langevin Equation (CLE). Alias for `noise_coupling`."""
        return self.noise_coupling(t, x, args)

    def diffrax_ode_term(self) -> ODETerm:
        """Returns an `ODETerm` for solving the system as an ODE with `diffrax`."""
        return ODETerm(self.vector_field)

    def diffusion_coeff_matrix(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The diffusion coefficient matrix for the Fokker-Planck equation.

        This matrix, `D(t, X_t)`, is defined as `g @ g.T`, where `g` is the
        noise coupling matrix from the CLE. It is equivalent to
        `S @ diag(a(X_t, t)) @ S.T`.

        Note:
            This is different from `noise_coupling`, which is the `g` matrix itself.

        Args:
            t: The current time.
            x: The current state vector (species counts).
            args: Additional arguments (not used).

        Returns:
            The diffusion coefficient matrix of shape `(n_species, n_species)`.
        """
        noise_coupling = self.noise_coupling(t, x, args)

        # Diffusion coefficient matrix: S * diag(a(X_t, t)) * S^T = g(t, X_t) * g(t, X_t)^T
        diffusion_matrix = noise_coupling @ noise_coupling.T

        return diffusion_matrix

    def diffusion_coeff(
        self, t: jnp.floating, x: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """The diffusion coefficient for the Fokker-Planck equation. Alias for `diffusion_coeff_matrix`."""
        return self.diffusion_coeff_matrix(t, x, args)

    def diffrax_sde_term(
        self,
        t1: float,
        t0: float = 0.0,
        tol: float = 1e-3,
        *,
        key: jnp.ndarray = None,
    ) -> MultiTerm:
        """Returns a `MultiTerm` for solving the system as an SDE with `diffrax`.

        This configures the SDE with a drift term (from the ODE vector field)
        and a diffusion term, controlled by a `VirtualBrownianTree`.

        Args:
            t1: The end time of the simulation.
            t0: The start time of the simulation. Defaults to 0.0.
            tol: The tolerance for the Brownian motion simulation.
            key: The `jax.random.PRNGKey` for the Brownian motion.

        Returns:
            A `diffrax.MultiTerm` for SDE integration.
        """
        return MultiTerm(
            ODETerm(self.drift_fn),
            ControlTerm(
                self.noise_coupling,
                VirtualBrownianTree(
                    t0,
                    t1,
                    shape=(self.n_reactions,),
                    tol=tol,
                    key=key,
                ),
            ),
        )

    # --- Likelihood computation ---

    def log_prob(self, sim_results: SimulationResults) -> jnp.ndarray:
        """Calculates the log-probability of a simulation trajectory.

        This method computes the log-likelihood of a given trajectory produced
        by an exact stochastic simulation algorithm (e.g., Gillespie Direct Method). It
        recomputes the propensities at each step of the trajectory to ensure
        that the computation graph is connected for gradient-based optimization.

        Args:
            sim_results: A `SimulationResults` object containing the trajectory.

        Returns:
            Log-probability terms, one for each step in the trajectory.
        """
        # Recompute propensities to build the JAX computation graph for gradients.
        # We operate on x[:-1] because the last state has no subsequent reaction.

        x = pytree_to_state(sim_results.x, self.species)
        t = sim_results.t
        propensities = jax.vmap(self.propensity_fn)(x[:-1], t[:-1])

        reactions = sim_results.reactions
        is_valid_reaction = reactions >= 0

        # Total propensity at each time step
        a0 = propensities.sum(axis=1)

        # Select the propensity of the reaction that occurred at each time step. Uses advanced indexing for efficiency.
        num_timesteps = propensities.shape[0]
        rows = jnp.arange(num_timesteps)

        # Clip indices to be valid for the gather operation. The results for
        # padded steps (where `is_valid_reaction` is False) are irrelevant
        # as they will be masked out later.
        safe_reactions = jnp.maximum(reactions, 0)
        a_selected_potentially_invalid = propensities[rows, safe_reactions]

        # Calculate log of selected propensity, ensuring numerical stability.
        # For invalid/padded steps, log(1.0) = 0.0, so they do not contribute
        # to the final log-probability.
        tiny = jnp.finfo(propensities.dtype).tiny
        log_a_selected = jnp.log(
            jnp.where(
                is_valid_reaction,
                jnp.maximum(a_selected_potentially_invalid, tiny),
                1.0,
            )
        )

        # Calculate the second term of the log-likelihood: `tau * a0`.
        # Mask out invalid steps by multiplying by the boolean mask.
        tau = jax.lax.stop_gradient(jnp.diff(sim_results.t))

        return log_a_selected - tau * a0 * is_valid_reaction

    # --- Utility methods ---

    def copy(self) -> ReactionNetwork:
        """Creates a new `ReactionNetwork` instance from the current one.

        This method provides a way to create a mutable copy of the network.
        It reconstructs the network from its reactions and volume, which is
        more robust and efficient than a generic deep copy.

        Returns:
            A new `ReactionNetwork` instance.
        """
        return ReactionNetwork(self.reactions, self.volume)

    def __add__(
        self, other: Reaction | Iterable[Reaction] | ReactionNetwork
    ) -> ReactionNetwork:
        """Adds reactions to the network.

        This is a functional operation that returns a new `ReactionNetwork`
        instance with the added reaction(s). It supports adding a single
        `Reaction`, another `ReactionNetwork`, or an iterable of `Reaction` objects.

        When adding another `ReactionNetwork`, the volume of the left-hand
        operand is used for the new network.

        Args:
            other: The reaction or reactions to add.

        Returns:
            A new `ReactionNetwork` with the added reaction(s).
        """
        if isinstance(other, Reaction):
            new_reactions = self.reactions + (other,)
        elif isinstance(other, ReactionNetwork):
            new_reactions = self.reactions + other.reactions
        elif isinstance(other, Iterable) and not isinstance(other, str):
            new_reactions = self.reactions + tuple(other)
        else:
            return NotImplemented

        return ReactionNetwork(new_reactions, volume=self.volume)

    def __getattr__(self, name: str) -> Reaction:
        """Access a reaction by its name.

        Allows retrieving a reaction from the network using attribute-style
        access, e.g., `network.reaction_name`.

        Args:
            name: The name of the reaction to retrieve.

        Returns:
            The `Reaction` object with the specified name.

        Raises:
            AttributeError: If no reaction with the given name is found.
        """
        if name in self._named_reactions:
            # Get the index of the substep from the lightweight dictionary
            index = self._named_reactions[name]
            # Return the substep from the current instance's tuple
            return self.reactions[index]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, i: int | slice | list[str]) -> Reaction | ReactionNetwork:
        """Access reactions by index, slice, or name.

        - If `i` is an integer, returns the `Reaction` at that index.
        - If `i` is a slice, returns a new `ReactionNetwork` with the sliced reactions.
        - If `i` is a list of strings, returns a new `ReactionNetwork` containing
          the reactions with the specified names.

        Args:
            i: The index, slice, or list of names to access.

        Returns:
            A single `Reaction` or a new `ReactionNetwork`.

        Raises:
            TypeError: If indexing is attempted with an unsupported type.
        """
        if isinstance(i, int):
            return self.reactions[i]
        elif isinstance(i, slice):
            return ReactionNetwork(self.reactions[i], volume=self.volume)
        elif isinstance(i, list) and all(isinstance(name, str) for name in i):
            return ReactionNetwork(
                [self.reactions[self._named_reactions[name]] for name in i],
                volume=self.volume,
            )
        else:
            raise TypeError(f'Indexing with type {type(i)} is not supported')

    def __iter__(self):
        """Iterate over the reactions in the network."""
        yield from self.reactions

    def __len__(self):
        """Return the number of reactions in the network."""
        return len(self.reactions)

    def __str__(self) -> str:
        """Generates a human-readable string representation of the network."""
        if not self.reactions:
            return ''

        reactant_matrix = jnp.asarray(self.reactant_matrix)
        product_matrix = jnp.asarray(self.product_matrix)

        # Prepare all parts for formatting
        parts = []
        for i, reaction in enumerate(self.reactions):
            reactants = ' + '.join(
                (
                    f'{int(value)} {self.species[j]}'
                    if value > 1
                    else f'{self.species[j]}'
                )
                for j, value in enumerate(reactant_matrix[:, i])
                if value > 0
            )
            products = ' + '.join(
                (
                    f'{int(value)} {self.species[j]}'
                    if value > 1
                    else f'{self.species[j]}'
                )
                for j, value in enumerate(product_matrix[:, i])
                if value > 0
            )
            reactants = reactants if reactants else '0'
            products = products if products else '0'
            kinetics_str = f'{type(reaction.kinetics).__name__}'
            reaction_str = f'{reactants} -> {products}'
            reaction_naming = f'R{i}'
            if reaction.name:
                reaction_naming = f'{reaction_naming} ({reaction.name})'

            parts.append((f'{reaction_naming}:', reaction_str, kinetics_str))

        # Find max lengths for alignment
        max_naming_len = max(len(p[0]) for p in parts)
        max_reaction_len = max(len(p[1]) for p in parts)

        # Build final descriptions
        descriptions = []
        for naming, reaction, kinetics in parts:
            padded_naming = naming.ljust(max_naming_len)
            padded_reaction = reaction.ljust(max_reaction_len)
            descriptions.append(f'{padded_naming}  {padded_reaction}  |  {kinetics}')

        return '\n'.join(descriptions)

    def to_latex(self, print_kinetics: bool = False) -> str:
        """Generates a LaTeX representation of the reaction network.

        This method produces a LaTeX string suitable for rendering in reports,
        notebooks, or publications. It automatically pairs forward and reverse
        reactions into reversible reactions.

        Args:
            print_kinetics: Whether to include the kinetics types in the output.

        Returns:
            A string containing the LaTeX representation of the reaction network, wrapped in `$$...$$` and an `align*` environment.
        """
        reactant_matrix = jnp.asarray(self.reactant_matrix)
        product_matrix = jnp.asarray(self.product_matrix)
        n_reactions = len(self.reactions)
        processed_indices = set()
        descriptions = []

        def _format_side_latex(coeffs, species_names):
            parts = []
            for k, value in enumerate(coeffs):
                if value > 0:
                    # Escape underscores for LaTeX rendering.
                    species_name_latex = species_names[k].replace('_', r'\_')
                    if value == 1:
                        parts.append(species_name_latex)
                    else:
                        # Use \, for a small space between coefficient and species
                        parts.append(f'{int(value)}\\,{species_name_latex}')
            side_str = ' + '.join(parts)
            return side_str if side_str else r'\emptyset'

        # O(n) optimization: Build a hash map of reaction signatures to indices
        # This avoids the O(n²) nested loop for finding reverse reactions
        reaction_signatures = {}
        for i in range(n_reactions):
            # Create a signature tuple: (reactants_tuple, products_tuple)
            # .tolist() converts from JAX array to python list, then tuple() makes it hashable.
            reactants_sig = tuple(reactant_matrix[:, i].tolist())
            products_sig = tuple(product_matrix[:, i].tolist())
            signature = (reactants_sig, products_sig)

            if signature not in reaction_signatures:
                reaction_signatures[signature] = []
            reaction_signatures[signature].append(i)

        # Process reactions, looking for reverse pairs using the hash map
        for i in range(n_reactions):
            if i in processed_indices:
                continue

            # Look for reverse reaction using hash map lookup (O(1))
            # .tolist() converts from JAX array to python list, then tuple() makes it hashable.
            reactants_sig = tuple(reactant_matrix[:, i].tolist())
            products_sig = tuple(product_matrix[:, i].tolist())
            reverse_signature = (
                products_sig,
                reactants_sig,
            )  # Swap reactants and products

            reverse_j = -1
            if reverse_signature in reaction_signatures:
                # Find a reverse reaction that hasn't been processed yet
                for candidate_j in reaction_signatures[reverse_signature]:
                    if candidate_j != i and candidate_j not in processed_indices:
                        reverse_j = candidate_j
                        break

            reactants = _format_side_latex(reactant_matrix[:, i], self.species)
            products = _format_side_latex(product_matrix[:, i], self.species)

            if reverse_j != -1:
                processed_indices.add(reverse_j)
                arrow = '\\leftrightarrow'
                reaction_str = f'{reactants} {arrow} {products}'

                if print_kinetics:
                    k_fwd = type(self.reactions[i].kinetics).__name__
                    k_rev = type(self.reactions[reverse_j].kinetics).__name__
                    if k_fwd == k_rev:
                        kinetics_str = f'\\text{{{k_fwd}}}'
                    else:
                        kinetics_str = f'\\text{{{k_fwd}}}/\\text{{{k_rev}}}'

                    descriptions.append(
                        f'{reaction_str} & \\quad ({kinetics_str}) \\\\'
                    )
                else:
                    descriptions.append(f'{reaction_str} \\\\')

            else:
                arrow = '\\rightarrow'
                reaction_str = f'{reactants} {arrow} {products}'
                if print_kinetics:
                    kinetics_str = (
                        f'\\text{{{type(self.reactions[i].kinetics).__name__}}}'
                    )
                    descriptions.append(
                        f'{reaction_str} & \\quad ({kinetics_str}) \\\\'
                    )
                else:
                    descriptions.append(f'{reaction_str} \\\\')

        description = (
            '$$\n\\begin{align*}\n' + '\n'.join(descriptions) + '\n\\end{align*}\n$$'
        )
        return description
