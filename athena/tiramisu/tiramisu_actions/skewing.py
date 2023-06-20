from __future__ import annotations
import itertools

from typing import Dict, TYPE_CHECKING, List, Tuple
from athena.tiramisu.compiling_service import CompilingService
from athena.tiramisu.tiramisu_program import TiramisuProgram

if TYPE_CHECKING:
    from athena.tiramisu.tiramisu_tree import TiramisuTree
from athena.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuActionType,
    TiramisuAction,
)


class Skewing(TiramisuAction):
    """
    Skewing optimization command.
    """

    def __init__(self, params: list, comps: list):
        # Skewing  takes four parameters of the 2 loops to skew and their factors
        assert len(params) == 4
        assert len(comps) > 0

        super().__init__(type=TiramisuActionType.SKEWING, params=params, comps=comps)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        self.tiramisu_optim_str = ""
        levels_with_factors = [
            str(tiramisu_tree.iterators[param].level) if index < 2 else str(param)
            for index, param in enumerate(self.params)
        ]
        for comp in self.comps:
            self.tiramisu_optim_str += (
                f"\n\t{comp}.skew({', '.join(levels_with_factors)});"
            )

        self.str_representation = f"S(L{levels_with_factors[0]},L{levels_with_factors[1]},{levels_with_factors[2]},{levels_with_factors[3]})"

    @classmethod
    def get_candidates(
        cls, program_tree: TiramisuTree
    ) -> Dict[str, List[Tuple[str, str]]]:
        candidates: Dict[str, List[Tuple[str, str]]] = {}

        candidate_sections = program_tree.get_candidate_sections()

        for root in candidate_sections:
            candidates[root] = []
            for section in candidate_sections[root]:
                # Only consider sections with more than one iterator
                if len(section) > 1:
                    # Get all possible combinations of 2 successive iterators
                    candidates[root].extend(list(itertools.pairwise(section)))
        return candidates

    @classmethod
    def get_factors(
        cls,
        loop_levels: List[int],
        current_schedule: List[TiramisuAction],
        tiramisu_program: TiramisuProgram,
        comps_skewed_loops: List[str],
    ) -> Tuple[int, int]:
        factors = CompilingService.call_skewing_solver(
            tiramisu_program,
            current_schedule,
            loop_levels=loop_levels,
            comps_skewed_loops=comps_skewed_loops,
        )
        if factors is not None:
            return factors
        else:
            raise ValueError("Skewing did not return any factors")
