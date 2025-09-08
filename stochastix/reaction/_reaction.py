"""Chemical reaction representation and parsing utilities."""

from __future__ import annotations

import re
from collections import defaultdict

import equinox as eqx

from ..kinetics import AbstractKinetics


def _parse_reaction_string(reaction_string: str) -> tuple[dict, dict]:
    """Parse a reaction string into reactant and product dictionaries.

    Args:
        reaction_string: The reaction string to parse, e.g., "A + 2B -> C".
            Supports "->" and "<-" as reaction direction indicators. Empty
            sides of a reaction can be represented by "0" or an empty string.

    Returns:
        A tuple of two dictionaries: one for reactants and one for products.
        Keys are species names and values are stoichiometric coefficients.

    Raises:
        ValueError: If the reaction string is invalid, contains a
            bidirectional arrow '<->', or includes an unparsable term.
    """
    # Check for bidirectional reaction first
    if '<->' in reaction_string:
        raise ValueError(
            'Bidirectional reactions (with <->) are not currently supported. Use separate forward and reverse reactions.'
        )

    # Split into reactants and products
    if '->' in reaction_string:
        reactants_str, products_str = reaction_string.split('->')
    elif '<-' in reaction_string:
        products_str, reactants_str = reaction_string.split('<-')
    else:
        raise ValueError(
            f"Invalid reaction string: '{reaction_string}'. Must contain '->' or '<-' separator."
        )

    def _parse_side(side_str: str):
        species_dict = defaultdict(float)
        if side_str.strip() == '0' or side_str.strip() == '':
            return species_dict
        for term in side_str.split('+'):
            term = term.strip()
            # Match coefficient and species name
            match = re.match(r'(\d*\.?\d*)\s*([A-Za-z_][A-Za-z0-9_]*)', term)
            if match and match.group(2):  # ensure species name is not empty
                coeff_str, species = match.groups()
                coeff = float(coeff_str) if coeff_str else 1.0
                species_dict[species] += coeff
            elif re.match(
                r'^[A-Za-z_][A-Za-z0-9_]*$', term
            ):  # Handle species with no coefficient
                species_dict[term] += 1.0
            elif term:
                raise ValueError(f"Cannot parse term '{term}' in reaction string.")
        return dict(species_dict)

    return _parse_side(reactants_str), _parse_side(products_str)


class Reaction(eqx.Module):
    """An `equinox.Module` representing a single reaction channel.

    This class encapsulates a single, unidirectional chemical reaction. It
    parses a reaction string to identify reactants and products, and associates
    the reaction with a given kinetic law.

    Attributes:
        kinetics: The `AbstractKinetics` object governing the reaction rate.
        reaction_string: The string representation of the reaction.
        name: The optional name of the reaction.
        reactants_and_coeffs: A tuple of (species, coefficient) pairs for the
            reactants.
        products_and_coeffs: A tuple of (species, coefficient) pairs for the
            products.
        kinetics_species: A tuple of species names that the kinetic law
            depends on, beyond the reactants.
        reactants: The reactant species of the reaction.
        products: The product species of the reaction.
        required_species: All species involved in the reaction.
    """

    kinetics: AbstractKinetics
    reaction_string: str = eqx.field(static=True)
    name: str = eqx.field(static=True, default=None)
    reactants_and_coeffs: tuple[tuple[str, float], ...] = eqx.field(static=True)
    products_and_coeffs: tuple[tuple[str, float], ...] = eqx.field(static=True)
    kinetics_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        reaction_string: str,
        kinetics: AbstractKinetics,
        name: str | None = None,
    ):
        """Initialize the Reaction.

        Args:
            reaction_string: The reaction string, e.g., "A + B -> C".
            kinetics: The kinetic law for the reaction.
            name: An optional name for the reaction.
        """
        self.reaction_string = reaction_string
        self.kinetics = kinetics
        self.name = name

        reactants, products = _parse_reaction_string(reaction_string)

        self.reactants_and_coeffs = tuple(sorted(reactants.items()))
        self.products_and_coeffs = tuple(sorted(products.items()))
        self.kinetics_species = tuple(sorted(kinetics._requires_species))

    @property
    def reactants(self) -> tuple[str, ...]:
        """The reactant species of the reaction."""
        return tuple(sorted(species for species, _ in self.reactants_and_coeffs))

    @property
    def products(self) -> tuple[str, ...]:
        """The product species of the reaction."""
        return tuple(sorted(species for species, _ in self.products_and_coeffs))

    @property
    def required_species(self) -> tuple[str, ...]:
        """All species involved in the reaction.

        This includes reactants, products, and any other species that the
        kinetic law depends on (e.g., regulators in Hill kinetics).
        """
        all_species = set(self.reactants)
        all_species.update(self.products)
        all_species.update(self.kinetics_species)
        return tuple(sorted(all_species))
