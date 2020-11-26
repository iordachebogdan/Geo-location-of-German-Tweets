import argparse
from datetime import datetime
import json
import os

from lib.utils.config_utils import expand_config

parser = argparse.ArgumentParser(description="Classic ML algorithms for regression")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)


def main():
    args = parser.parse_args()
    config = {}
    with open(args.config) as f:
        config = json.load(f)

    configs = expand_config(config)
    results_path = (
        f'runs/{datetime.now()}-{config["type"]}-{config["method"]}'
        f'-{config["algorithm"]}'
    )
    os.mkdir(results_path)
    for i, config in enumerate(configs):
        current_results_path = os.path.join(results_path, str(i))
        os.mkdir(current_results_path)
        with open(current_results_path + "/config.json", "w") as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
