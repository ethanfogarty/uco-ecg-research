import itertools
import multiprocessing as mp

from Scripts.brno_cnnvae_gs_dir.singletrain import TrainOne

# ------------------------------------------------------------------
#                      Create Param Grid  --------------------------
# ------------------------------------------------------------------
param_space = {
    "latent_dim": [300],
    "unit_count": [[32, 64, 128],
                   [64, 128, 256],
                   [128, 256, 512]],
    "layer_count": [3]
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
    runId = 114
    for paramCombo in combos:
        ctx = mp.get_context("forkserver")
        p = ctx.Process(target=TrainOne, args=(runId, paramCombo))
        p.start()
        p.join()
        #if p.exitcode != 0:
        #    raise RuntimeError(f"run {runId} crashed (exit {p.exitcode})")
        p.close()
        runId += 1