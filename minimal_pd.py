import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

from env import run_sim
import time

# an idea can trigger the explosion of a universe.
if __name__ == "__main__":
    start = time.time()
    verbose = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
    run_sim(steps=200000, N=50, history_len=512, p_death=0.002, log_every=100, out_csv="sim_log_new.csv", pairs_per_step=10, train_every=10, verbose=verbose)
    print("done", time.time() - start)
