from collections import defaultdict
from typing import NamedTuple

import numpy as np


class AircraftTypeDict(NamedTuple):
    # Aircraft code
    # Using a string to be more generic, really it does not matter
    # TODO: what code is this?
    code: str

    # Name: a displayable aircraft type name
    name: str

    # Age distribution for this type, from youngest (index 0) to oldest (index -1)
    # shape = (num_age_bins)
    age_distribution: np.ndarray

    # ASMs that this aircraft type flew during the initial year, broken down by mission
    # Keys are mission names, missing mission is assumed to be 0
    asm_by_mission: dict[str, float]

    # Survival curve tha described the fraction of initial aircraft remaining after n years
    # Starts at 1 and ends at 0
    # The retirement curve obtained from the survival curve corresponds to the fraction that does not survive between year i and year (i + 1)
    survival_curve: np.ndarray

    # The energy intensity of this type relative to an initial year's reference aircraft
    # 1 means same intensity, >1 means more energy per ASM, <1 means less energy per ASM
    relative_energy_intensity: float

    # The breakdown of energy sources for this type
    # Must sum to 1, missing energy carriers are assumed to be 0
    energy_fractions_per_carrier: dict[str, float]

    # List of other types that this type replaces
    # Replacements can be customized on a per-mission basis (e.g., a type can only replace
    # another type for one of its missions). For the moment, a type can only be replaced by a
    # single type per mission and the code will throw an error is this is not respected. This
    # is because allowig for multiple replacements would also require specifying replacement
    # ratios.
    # Specified as a dictionary where keys indicate the mission and the value is a tuple
    # whose first element is the first year at which that replacement occurs, and the second
    # is the ID of the replaced type
    replaces: dict[str, tuple[int, str]]


def create_initial_asm_fractions(types, missions, types_dict: list[AircraftTypeDict]):
    # TODO: need to compute ASM fraction from total ASM

    # ASM by type
    types_asm = {
        type_code: np.array(list(types_dict[type_code].asm_by_mission.values())).sum()
        for type_code in types
    }

    # Total ASM
    total_asm = np.array(list(types_asm.values())).sum()

    # Create the ASM fraction array for each type
    type_arrays = []
    for type_code in types:
        type_dict = types_dict[type_code]
        age_distribution = type_dict.age_distribution

        per_mission_asm_fractions = np.array(
            [
                (
                    type_dict.asm_by_mission[mission] / types_asm[type_code]
                    if types_asm[type_code] > 0
                    else 0
                )
                if mission in type_dict.asm_by_mission
                else 0
                for mission in missions
            ]
        )
        type_asm_fraction = types_asm[type_code] / total_asm

        type_asm_fractions = (
            age_distribution[np.newaxis, :]
            * per_mission_asm_fractions[:, np.newaxis]
            * type_asm_fraction
        )

        type_arrays.append(type_asm_fractions)

    initial_asm_fractions = np.stack(type_arrays, axis=0)

    return initial_asm_fractions

    # return np.stack(
    #     [
    #         types_dict[type_]["age_distribution"][np.newaxis, :]
    #         * np.array(
    #             [
    #                 types_dict[type_]["per_mission_asm_fractions"][mission]
    #                 if mission in types_dict[type_]["per_mission_asm_fractions"]
    #                 else 0
    #                 for mission in missions
    #             ]
    #         )[:, np.newaxis]
    #         * types_dict[type_]["fraction_of_total_asm"]
    #         for type_ in types
    #     ],
    #     axis=0,
    # )


def create_retirement_curves(types, types_dict):
    # retirement proportions for all age bins except the last one
    all_but_last_age_bin = np.stack(
        [
            1
            - types_dict[type_code].survival_curve[1:]
            / (types_dict[type_code].survival_curve[:-1] + 1e-6)
            for type_code in types
        ],
        axis=0,
    )

    # all remaining aircraft in the last year bin get retired
    return np.concatenate((all_but_last_age_bin, np.ones((len(types), 1))), axis=1)


def create_relative_energy_intensities(types, types_dict):
    return np.array(
        [types_dict[type_code].relative_energy_intensity for type_code in types]
    )


def create_energy_carrier_breakdown(types, energy_carriers, types_dict):
    return np.stack(
        [
            np.array(
                [
                    types_dict[type_code].energy_fractions_per_carrier[energy_carrier]
                    if energy_carrier
                    in types_dict[type_code].energy_fractions_per_carrier
                    else 0
                    for energy_carrier in energy_carriers
                ]
            )
            for type_code in types
        ],
        axis=0,
    )


def get_type_replacement_matrix(
    type_code,
    types_index,
    types_dict,
    replacement_matrices_dict,
    successors_dict,
    num_years,
    num_ramp_years,
    start_year,
    missions,
):
    # If this type's matrix has already been prepared before
    if type_code in replacement_matrices_dict:
        return replacement_matrices_dict[type_code]

    # By default, a type replaces itself
    type_replacement_matrix = np.zeros((num_years, len(missions), len(types_dict)))
    type_replacement_matrix[:, :, types_index[type_code]] = 1

    # If it has a successor, we blend the type's replacement matrix with its successor's
    for mission_index, mission in enumerate(missions):
        # If the current type does not have a replacement for this mission
        if mission not in successors_dict[type_code]:
            continue

        # Retrieve successor's year and type
        successor_year, successor_type = successors_dict[type_code][mission]

        # Retrieve's the successor's replacement matrix
        successor_replacement_matrix = get_type_replacement_matrix(
            successor_type,
            types_index,
            types_dict,
            replacement_matrices_dict,
            successors_dict,
            num_years,
            num_ramp_years,
            start_year,
            missions,
        )

        # The index along the years dimension at which to initiate the replacement
        replacement_start_year_index = successor_year - start_year
        replacement_end_ramp_year_index = replacement_start_year_index + num_ramp_years

        # Clipping to bounds
        replacement_start_year_index = max(
            0, min(replacement_start_year_index, num_years - 1)
        )
        replacement_end_ramp_year_index = max(
            0, min(replacement_end_ramp_year_index, num_years - 1)
        )

        # Matrix used to blend with successor's replacement matrix
        actual_ramp_duration = (
            replacement_end_ramp_year_index - replacement_start_year_index + 1
        )
        full_replacement_duration = num_years - replacement_end_ramp_year_index - 1
        blending_vector = np.concatenate(
            [
                np.ones((replacement_start_year_index,)),
                np.linspace(1, 0, actual_ramp_duration),
                np.zeros((full_replacement_duration,)),
            ]
        )

        type_replacement_matrix[:, mission_index, :] = np.einsum(
            "jk,jl->lk",
            type_replacement_matrix[:, mission_index, :],
            np.diag(blending_vector),
        ) + np.einsum(
            "jk,jl->lk",
            successor_replacement_matrix[:, mission_index, :],
            np.diag(1 - blending_vector),
        )

    # Store the prepared replacement matrix to avoid duplicating work
    replacement_matrices_dict[type_code] = type_replacement_matrix

    return type_replacement_matrix


def create_successors_dict(types_dict):
    # Dictionary that maps a type to its successors on a per-mission basis
    successors_dict = defaultdict(dict)
    for type_, type_dict in types_dict.items():
        for mission, replacements in type_dict.replaces.items():
            for replacement_year, replaced_type in replacements:
                if (
                    replaced_type in successors_dict
                    and mission in successors_dict[replaced_type]
                ):
                    raise RuntimeError(
                        "A type can only have a single replacement per mission."
                    )
                successors_dict[replaced_type][mission] = (replacement_year, type_)
    return successors_dict


def create_replacement_matrix(
    types,
    types_dict,
    num_years,
    num_ramp_years,
    start_year,
    missions,
):
    # Holds replacement matrices for each type to avoid repeated computations during the recursion
    replacement_matrices_dict = {}

    # Maps a type to its index
    types_index = {type_: type_index for type_index, type_ in enumerate(types)}

    # Maps a type to its successors for each mission
    successors_dict = create_successors_dict(types_dict)

    # Get each type's replacement matrix
    return np.stack(
        [
            get_type_replacement_matrix(
                type_,
                types_index,
                types_dict,
                replacement_matrices_dict,
                successors_dict,
                num_years,
                num_ramp_years,
                start_year,
                missions,
            )
            for type_ in types
        ],
        axis=2,
    )
