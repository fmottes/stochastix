"""Pre-built reaction network generators for common biological systems."""

from __future__ import annotations

import jax.numpy as jnp

from .kinetics import HillAA, HillActivator, HillRepressor, MassAction
from .reaction import Reaction, ReactionNetwork


def lotka_volterra_model(
    alpha: float | jnp.floating = 1.1,
    beta: float | jnp.floating = 0.4,
    gamma: float | jnp.floating = 0.4,
) -> ReactionNetwork:
    """Generate a basic Lotka-Volterra predator-prey model.

    This model describes the dynamics of two interacting species, a predator and
    a prey, based on the following reactions:

    1.  Prey reproduction: `prey -> 2 prey` with rate `alpha`.
    2.  Predator-prey interaction: `prey + predator -> 2 predator` with rate `beta`.
    3.  Predator death: `predator -> 0` with rate `gamma`.

    Args:
        alpha: The rate of prey reproduction.
        beta: The rate of predation.
        gamma: The rate of predator death.

    Returns:
        ReactionNetwork representing the Lotka-Volterra system.
    """
    reactions = [
        Reaction('prey -> 2 prey', MassAction(alpha), name='prey_reproduction'),
        Reaction('prey + predator -> 2 predator', MassAction(beta), name='predation'),
        Reaction('predator -> 0', MassAction(gamma), name='predator_death'),
    ]
    return ReactionNetwork(reactions)


def sirs_model(
    beta: float | jnp.floating = 1.0,
    gamma: float | jnp.floating = 0.1,
    nu: float | jnp.floating = 0.01,
) -> ReactionNetwork:
    """Generate a classic SIRS epidemiological model.

    This model describes the spread of an infectious disease in a population
    based on the following reactions:

    1.  Infection: `S + I -> 2 I` (transmission rate `beta`)
    2.  Recovery: `I -> R` (recovery rate `gamma`)
    3.  Loss of immunity: `R -> S` (waning immunity rate `nu`)

    The model assumes:
    -   S: Susceptible individuals
    -   I: Infected individuals
    -   R: Recovered individuals (with temporary immunity)

    Note:
        Setting the waning immunity rate `nu` to zero reduces this model to the
        classic SIR model.

    Args:
        beta: The transmission rate (rate of infection per S-I contact).
        gamma: The recovery rate (rate at which infected individuals recover).
        nu: The waning immunity rate (rate at which recovered individuals
            lose immunity and become susceptible again).

    Returns:
        ReactionNetwork representing the SIRS system.
    """
    reactions = [
        Reaction('S + I -> 2 I', MassAction(beta), name='infection'),
        Reaction('I -> R', MassAction(gamma), name='recovery'),
        Reaction('R -> S', MassAction(nu), name='loss_of_immunity'),
    ]
    return ReactionNetwork(reactions)


def single_gene_expression_model(
    k_m: float | jnp.floating = 0.1,
    k_p: float | jnp.floating = 10.0,
    gamma_m: float | jnp.floating = 0.1,
    gamma_p: float | jnp.floating = 0.01,
) -> ReactionNetwork:
    """Generate a Thattai-van Oudenaarden model for single-gene expression.

    This model describes the stochastic production and degradation of mRNA and
    protein from a single gene. It includes four fundamental reactions:

    1.  Transcription: `0 -> mRNA` (mRNA production)
    2.  Translation: `mRNA -> mRNA + P` (Protein production from mRNA)
    3.  mRNA Degradation: `mRNA -> 0`
    4.  Protein Degradation: `P -> 0`

    Args:
        k_m: Transcription rate (rate of mRNA production).
        k_p: Translation rate (rate of protein production per mRNA).
        gamma_m: mRNA degradation rate.
        gamma_p: Protein degradation rate.

    Returns:
        ReactionNetwork representing the single-gene expression system.
    """
    reactions = [
        Reaction('0 -> mRNA', MassAction(k_m), name='transcription'),
        Reaction('mRNA -> mRNA + P', MassAction(k_p), name='translation'),
        Reaction('mRNA -> 0', MassAction(gamma_m), name='mrna_degradation'),
        Reaction('P -> 0', MassAction(gamma_p), name='protein_degradation'),
    ]
    return ReactionNetwork(reactions)


def michaelis_menten_explicit_model(
    k_f: float | jnp.floating = 0.01,
    k_r: float | jnp.floating = 0.001,
    k_cat: float | jnp.floating = 0.1,
) -> ReactionNetwork:
    """Generate an explicit Michaelis-Menten enzyme kinetics model.

    This model explicitly describes the formation of an enzyme-substrate
    complex (C) from an enzyme (E) and substrate (S), and the subsequent
    production of a product (P).

    The model includes three reactions:
    1.  Binding: `E + S -> C` (forward reaction)
    2.  Unbinding: `C -> E + S` (reverse reaction)
    3.  Conversion: `C -> E + P` (catalytic reaction)

    Args:
        k_f: The forward rate constant for enzyme-substrate binding.
        k_r: The reverse rate constant for complex dissociation.
        k_cat: The catalytic rate constant for product formation.

    Returns:
        ReactionNetwork representing the explicit Michaelis-Menten system.
    """
    reactions = [
        Reaction('E + S -> C', MassAction(k_f), name='binding'),
        Reaction('C -> E + S', MassAction(k_r), name='unbinding'),
        Reaction('C -> E + P', MassAction(k_cat), name='conversion'),
    ]
    return ReactionNetwork(reactions)


def schlogl_model(
    k1: float | jnp.floating = 1.0,
    k2: float | jnp.floating = 0.18,
    k3: float | jnp.floating = 2.4,
    k4: float | jnp.floating = 1.0,
) -> ReactionNetwork:
    """Generate the Schlögl model, a classic example of a bistable system.

    This model describes the dynamics of a single species X subject to
    autocatalysis and degradation. The reactions are:
    1.  Autocatalysis: `2X -> 3X` (forward reaction)
    2.  Reverse Autocatalysis: `3X -> 2X` (reverse reaction)
    3.  Production: `0 -> X` (zeroth-order production)
    4.  Degradation: `X -> 0` (first-order degradation)

    The model is known to exhibit bistability for certain parameter values,
    meaning it can exist in two different stable steady states.

    Args:
        k1: The rate constant for the forward autocatalytic reaction.
        k2: The rate constant for the reverse autocatalytic reaction.
        k3: The rate constant for the production reaction.
        k4: The rate constant for the degradation reaction.

    Returns:
        ReactionNetwork representing the Schlögl system.
    """
    reactions = [
        Reaction('2X -> 3X', MassAction(k1), name='autocatalysis'),
        Reaction('3X -> 2X', MassAction(k2), name='reverse_autocatalysis'),
        Reaction('0 -> X', MassAction(k3), name='production'),
        Reaction('X -> 0', MassAction(k4), name='degradation'),
    ]
    return ReactionNetwork(reactions)


def hopfield_kinetic_proofreading_model(
    k_f: float | jnp.floating = 1.0,
    k_r: float | jnp.floating = 0.1,
    k_p: float | jnp.floating = 1.0,
    k_d: float | jnp.floating = 0.1,
    k_cat: float | jnp.floating = 10.0,
) -> ReactionNetwork:
    """Generate a Hopfield kinetic proofreading model.

    This model describes how an enzyme can achieve high fidelity by introducing
    an intermediate, energy-dependent proofreading step. This allows the system
    to discard incorrectly bound substrates before they are converted to a product.

    The model involves a substrate (S), an enzyme (E), an initial complex (C),
    an activated complex (C_activated), and a product (P).

    The reactions are:
    1.  Binding: `E + S -> C` (rate `k_f`)
    2.  Unbinding: `C -> E + S` (rate `k_r`)
    3.  Activation (Proofreading step): `C -> C_activated` (rate `k_p`)
    4.  Discard: `C_activated -> E + S` (rate `k_d`)
    5.  Catalysis: `C_activated -> E + P` (rate `k_cat`)

    Args:
        k_f: The forward rate constant for enzyme-substrate binding.
        k_r: The reverse rate constant for initial complex dissociation.
        k_p: The rate of activation to the proofreading state.
        k_d: The rate of discarding the substrate from the activated state.
        k_cat: The catalytic rate for product formation.

    Returns:
        ReactionNetwork representing the Hopfield kinetic proofreading system.
    """
    reactions = [
        Reaction('E + S -> C', MassAction(k_f), name='binding'),
        Reaction('C -> E + S', MassAction(k_r), name='unbinding'),
        Reaction('C -> C_activated', MassAction(k_p), name='activation'),
        Reaction('C_activated -> E + S', MassAction(k_d), name='discard'),
        Reaction('C_activated -> E + P', MassAction(k_cat), name='catalysis'),
    ]
    return ReactionNetwork(reactions)


def toggle_switch_model(
    alpha: float | jnp.floating | jnp.ndarray | tuple | list = 100.0,
    alpha0: float | jnp.floating | jnp.ndarray | tuple | list = 0.1,
    K: float | jnp.floating | jnp.ndarray | tuple | list = 50.0,
    beta: float | jnp.floating | jnp.ndarray | tuple | list = 1.0,
    n: float | jnp.floating | jnp.ndarray | tuple | list = 2.0,
) -> ReactionNetwork:
    """Generate a toggle switch gene regulatory network.

    The toggle switch is a synthetic bistable network where two genes
    mutually repress each other (A represses B, and B represses A), creating a
    system with two stable states.

    The model includes the following reactions:
    1.  Gene A expression, repressed by B.
    2.  Gene B expression, repressed by A.
    3.  Protein degradation of A and B.

    Args:
        alpha: Maximum expression rate(s). Can be a single float or a
            tuple/list for individual rates `(alpha_A, alpha_B)`.
        alpha0: Leaky expression rate(s). Can be a single float or a
            tuple/list for individual rates `(alpha0_A, alpha0_B)`.
        K: Half-saturation constant(s) for repression. Can be a single
            float or a tuple/list `(K_A, K_B)`.
        beta: Protein degradation rate(s). Can be a single float or a
            tuple/list `(beta_A, beta_B)`.
        n: Hill coefficient(s) for repression. Must be positive. Can be a
            single float or a tuple/list `(n_A, n_B)`.

    Returns:
        ReactionNetwork representing the toggle switch system.
    """
    if isinstance(alpha, jnp.ndarray | tuple | list):
        alphaA, alphaB = alpha
    else:
        alphaA, alphaB = alpha, alpha

    if isinstance(alpha0, jnp.ndarray | tuple | list):
        alpha0A, alpha0B = alpha0
    else:
        alpha0A, alpha0B = alpha0, alpha0

    if isinstance(K, jnp.ndarray | tuple | list):
        KA, KB = K
    else:
        KA, KB = K, K

    if isinstance(beta, jnp.ndarray | tuple | list):
        betaA, betaB = beta
    else:
        betaA, betaB = beta, beta

    if isinstance(n, jnp.ndarray | tuple | list):
        nA, nB = n
    else:
        nA, nB = n, n

    reactions = [
        # Repressed gene expression
        Reaction(
            '0 -> A',
            HillRepressor('B', alphaA, KA, nA, v0=alpha0A),
            name='gene_A_expression',
        ),
        Reaction(
            '0 -> B',
            HillRepressor('A', alphaB, KB, nB, v0=alpha0B),
            name='gene_B_expression',
        ),
        # Protein degradation
        Reaction('A -> 0', MassAction(betaA), name='protein_A_degradation'),
        Reaction('B -> 0', MassAction(betaB), name='protein_B_degradation'),
    ]
    return ReactionNetwork(reactions)


def repressilator_model(
    alpha: float | jnp.floating | jnp.ndarray | tuple | list = 100.0,
    alpha0: float | jnp.floating | jnp.ndarray | tuple | list = 1.0,
    K: float | jnp.floating | jnp.ndarray | tuple | list = 50.0,
    beta: float | jnp.floating | jnp.ndarray | tuple | list = 5.0,
    n: float | jnp.floating | jnp.ndarray | tuple | list = 2.0,
) -> ReactionNetwork:
    """Generate a repressilator gene regulatory network.

    The repressilator is a synthetic genetic clock built from three genes that
    repress each other in a cycle (A represses B, B represses C, and C
    represses A), leading to oscillatory dynamics.

    The model includes the following reactions:
    1.  Gene A expression, repressed by C.
    2.  Gene B expression, repressed by A.
    3.  Gene C expression, repressed by B.
    4.  Degradation of proteins A, B, and C.

    The expression rates are modeled using repressive Hill kinetics with a basal
    (leaky) expression term.

    Args:
        alpha: Maximum expression rate for each gene. Can be a single float
            or a tuple/list `(alpha_A, alpha_B, alpha_C)`.
        alpha0: Leaky expression rate for each gene. Can be a single float or
            a tuple/list `(alpha0_A, alpha0_B, alpha0_C)`.
        K: Half-saturation constant for repression. Can be a single float
            or a tuple/list `(K_A, K_B, K_C)`.
        beta: Protein degradation rate. Can be a single float or a tuple/list
            `(beta_A, beta_B, beta_C)`.
        n: Hill coefficient for repression (must be positive). Can be a
            single float or a tuple/list `(n_A, n_B, n_C)`.

    Returns:
        ReactionNetwork representing the repressilator system.
    """
    if isinstance(alpha, jnp.ndarray | tuple | list):
        alphaA, alphaB, alphaC = alpha
    else:
        alphaA, alphaB, alphaC = alpha, alpha, alpha

    if isinstance(alpha0, jnp.ndarray | tuple | list):
        alpha0A, alpha0B, alpha0C = alpha0
    else:
        alpha0A, alpha0B, alpha0C = alpha0, alpha0, alpha0

    if isinstance(K, jnp.ndarray | tuple | list):
        KA, KB, KC = K
    else:
        KA, KB, KC = K, K, K

    if isinstance(beta, jnp.ndarray | tuple | list):
        betaA, betaB, betaC = beta
    else:
        betaA, betaB, betaC = beta, beta, beta

    if isinstance(n, jnp.ndarray | tuple | list):
        nA, nB, nC = n
    else:
        nA, nB, nC = n, n, n

    reactions = [
        # Repressed gene expression (repressed by the previous gene in the cycle)
        # A is repressed by C, B is repressed by A, C is repressed by B
        Reaction(
            '0 -> A',
            HillRepressor('C', alphaA, KA, nA, v0=alpha0A),
            name='gene_A_expression',
        ),
        Reaction(
            '0 -> B',
            HillRepressor('A', alphaB, KB, nB, v0=alpha0B),
            name='gene_B_expression',
        ),
        Reaction(
            '0 -> C',
            HillRepressor('B', alphaC, KC, nC, v0=alpha0C),
            name='gene_C_expression',
        ),
        # Protein degradation
        Reaction('A -> 0', MassAction(betaA), name='protein_A_degradation'),
        Reaction('B -> 0', MassAction(betaB), name='protein_B_degradation'),
        Reaction('C -> 0', MassAction(betaC), name='protein_C_degradation'),
    ]
    return ReactionNetwork(reactions)


def ffl_c1_model(
    # X parameters
    k_x: float | jnp.floating = 0.0,
    beta_x: float | jnp.floating = 0.0,
    # Y parameters
    alpha_y: float | jnp.floating = 100.0,
    alpha0_y: float | jnp.floating = 0.1,
    K_xy: float | jnp.floating = 50.0,
    n_xy: float | jnp.floating = 2.0,
    beta_y: float | jnp.floating = 1.0,
    # Z parameters
    alpha_z: float | jnp.floating = 100.0,
    alpha0_z: float | jnp.floating = 0.1,
    K_xz: float | jnp.floating = 50.0,
    K_yz: float | jnp.floating = 50.0,
    n_xz: float | jnp.floating = 2.0,
    n_yz: float | jnp.floating = 2.0,
    beta_z: float | jnp.floating = 1.0,
    # Logic
    logic: str = 'and',
    competitive_binding: bool = False,
) -> ReactionNetwork:
    """Generate a coherent type 1 feed-forward loop (C1-FFL) network.

    In a C1-FFL, a master regulator X activates a second regulator Y, and both
    X and Y jointly activate a target gene Z. This motif is known for its
    ability to act as a sign-sensitive delay element or a persistence detector.

    The network consists of three genes (X, Y, Z) with the following interactions:
    1.  X production and degradation (optional).
    2.  Y production, activated by X.
    3.  Z production, activated by both X and Y.
    4.  Degradation of Y and Z.

    The regulation of Z by X and Y can be configured with 'and' or 'or' logic.

    Note:
        The default model is built with no dynamics for X. You can include
        reactions for X by extending the model after it is generated.

    Example:
        To add a reaction where X is produced at a constant rate:
        ```python
        model = ffl_c1_model(k_x=10.0)
        ```
        To add a reaction where X production is activated by a signal S:
        ```python
        from stochastix.kinetics import HillActivator
        from stochastix.reaction import Reaction

        model = ffl_c1_model()
        x_production_reaction = Reaction(
            '0 -> X',
            HillActivator(regulator='S', v=100.0, K=50.0, n=2.0, v0=0.1),
            name='X_production'
        )
        model = model.add_reaction(x_production_reaction)
        ```

    Args:
        k_x: Production rate of X. If 0, the reaction `0 -> X` is not included.
        beta_x: Degradation rate of X. If 0, the reaction `X -> 0` is not included.
        alpha_y: Maximum production rate of Y (activated by X).
        alpha0_y: Leaky production rate of Y.
        K_xy: Half-saturation constant for X activating Y.
        n_xy: Hill coefficient for X activating Y.
        beta_y: Degradation rate of Y.
        alpha_z: Maximum production rate of Z (activated by X and Y).
        alpha0_z: Leaky production rate of Z.
        K_xz: Half-saturation constant for X activating Z.
        K_yz: Half-saturation constant for Y activating Z.
        n_xz: Hill coefficient for X activating Z.
        n_yz: Hill coefficient for Y activating Z.
        beta_z: Degradation rate of Z.
        logic: The logic for Z regulation by X and Y ('and' or 'or').
        competitive_binding: Whether X and Y bind competitively to Z's promoter.

    Returns:
        ReactionNetwork representing the C1-FFL system.
    """
    if logic not in ['and', 'or']:
        raise ValueError("logic must be either 'and' or 'or'")

    reactions = []

    if k_x > 0:
        reactions.append(Reaction('0 -> X', MassAction(k_x), name='X_production'))

    if beta_x > 0:
        reactions.append(Reaction('X -> 0', MassAction(beta_x), name='X_degradation'))

    reactions.extend(
        [
            Reaction(
                '0 -> Y',
                HillActivator('X', v=alpha_y, K=K_xy, n=n_xy, v0=alpha0_y),
                name='Y_production',
            ),
            Reaction('Y -> 0', MassAction(beta_y), name='Y_degradation'),
            Reaction(
                '0 -> Z',
                HillAA(
                    activator1='X',
                    activator2='Y',
                    v=alpha_z,
                    K1=K_xz,
                    K2=K_yz,
                    n1=n_xz,
                    n2=n_yz,
                    logic=logic,
                    competitive_binding=competitive_binding,
                    v0=alpha0_z,
                ),
                name='Z_production',
            ),
            Reaction('Z -> 0', MassAction(beta_z), name='Z_degradation'),
        ]
    )

    return ReactionNetwork(reactions)
