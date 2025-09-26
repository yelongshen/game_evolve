
# Configure logging to save all logs to a file with a timestamp and also print to console
import datetime
import logging

log_filename = f"simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

from env import run_sim
import time

# an idea can trigger the explosion of a universe.
if __name__ == "__main__":
    import sys
    import logging
    start = time.time()
    verbose = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
    # Parse device from command line argument, default to "cpu"
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    try:
        run_sim(steps=200000, N=50, history_len=512, p_death=0.002, log_every=100, out_csv="sim_log_new.csv", pairs_per_step=10, train_every=10, verbose=verbose, device=device)
        print("done", time.time() - start)
    finally:
        # Ensure all logs are flushed and file handlers closed, even on exception
        logging.shutdown()
