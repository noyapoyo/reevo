import os
import json
from .original.eoh import EOH
from .original.getParas import Paras
from .original import prob_rank, pop_greedy
from .problem_adapter import Problem

from utils.utils import init_client

class EoH:
    def __init__(self, cfg, root_dir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)
        self.scheduled_evals = 2 * self.cfg.pop_size
        self.evolution_rounds = max(0, (self.cfg.max_fe - self.scheduled_evals) // (4 * self.cfg.pop_size))
        self.scheduled_evals += self.evolution_rounds * 4 * self.cfg.pop_size

        self.paras = Paras() 
        self.paras.set_paras(method = "eoh",
                    # problem = "Not used", # Not used
                    # llm_api_endpoint = "api.openai.com",
                    llm_model = client,
                    ec_pop_size = self.cfg.pop_size,
                    # EoH's evolution schedule is batch-shaped. We keep the
                    # normal schedule below max_fe, then fill the leftover
                    # budget by re-evaluating the best solver.
                    ec_n_pop = self.evolution_rounds,
                    exp_output_path = "./",
                    exp_debug_mode = False,
                    eva_timeout=cfg.timeout)
        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        method = EOH(self.paras, self.problem, prob_rank, pop_greedy)

        best_code, best_code_path = method.run()
        remaining_evals = max(0, int(self.cfg.max_fe) - int(self.scheduled_evals))
        fill_objs = []
        if remaining_evals > 0:
            print(
                f"- EoH budget fill: scheduled_evals={self.scheduled_evals}, "
                f"max_fe={self.cfg.max_fe}, best_solver_re_evals={remaining_evals} -"
            )
            fill_objs = self.problem.batch_evaluate([best_code] * remaining_evals, self.evolution_rounds + 1)
            with open("eoh_budget_fill.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "max_fe": int(self.cfg.max_fe),
                        "scheduled_evals": int(self.scheduled_evals),
                        "best_solver_re_evals": int(remaining_evals),
                        "total_evals_after_fill": int(self.scheduled_evals + remaining_evals),
                        "fill_objectives": [float(x) for x in fill_objs],
                    },
                    f,
                    indent=2,
                )

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        return best_code, best_code_path
