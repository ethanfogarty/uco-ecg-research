import itertools
import multiprocessing as mp

from Scripts.brno_classification.ffnn.singletrain import TrainOne

# ------------------------------------------------------------------
#                      Create Param Grid  --------------------------
# ------------------------------------------------------------------
param_space = {
    "layer_count": [10],
    "unit_count": [[200, 250, 300, 350, 400, 450, 500, 550, 600, 650]]
}
# every combination as a list of tuples
combos = list(itertools.product(*param_space.values()))

if __name__ == "__main__":
    # ------------------------------------------------------------------
    #                      Set Linux Start Method  ---------------------
    # ------------------------------------------------------------------
    mp.set_start_method("forkserver", force=True)
    #ctx = mp.get_context("forkserver")

    # ------------------------------------------------------------------
    #     Launch one worker per VAE configuration  ---------------------
    # ------------------------------------------------------------------
    runId = 23
    for paramCombo in combos:
        ctx = mp.get_context("forkserver")
        p = ctx.Process(target=TrainOne, args=(runId, paramCombo))
        p.start()
        p.join()
        #if p.exitcode != 0:
        #    raise RuntimeError(f"run {runId} crashed (exit {p.exitcode})")
        p.close()
        runId += 1