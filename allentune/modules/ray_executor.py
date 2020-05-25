import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

import ray
from ray.tune import function, register_trainable, run_experiments

from allentune import MULTI_PARAM_SEARCH, SINGLE_PARAM_SEARCH
from allentune.modules.allennlp_runner import AllenNlpRunner
from allentune.util.random_search import RandomSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RayExecutor(object):
    name = "Ray"

    def __init__(self, runner: AllenNlpRunner) -> None:
        self._runner = runner

    def parse_search_config(self, search_config: Dict) -> Dict:
        for hyperparameter, val in search_config.items():
            if not isinstance(val, dict):
                ray_sampler = val
            elif val['sampling strategy'] == 'loguniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_loguniform(low, high))
            elif val['sampling strategy'] == 'integer':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_integer(low, high))
            elif val['sampling strategy'] == 'choice':
                ray_sampler = function(RandomSearch.random_choice(val['choices']))
            elif val['sampling strategy'] == 'uniform':
                low, high = val['bounds'][0], val['bounds'][1]
                ray_sampler = function(RandomSearch.random_uniform(low, high))
            else:
                raise KeyError(f"sampling strategy {val['sampling strategy']} does not exist")
            search_config[hyperparameter] = ray_sampler
        return search_config

    def _build_single_run_config(self, args: argparse.Namespace) -> Dict:
        run_func = self._runner.get_single_run_func(args)
        register_trainable("run", run_func)

        with open(args.search_space) as f:
            search_config = json.load(f)
        search_config = self.parse_search_config(search_config)

        return {
            args.experiment_name: {
                "run": "run",
                "resources_per_trial": {
                    "cpu": args.cpus_per_trial,
                    "gpu": args.gpus_per_trial,
                },
                "config": search_config,
                "local_dir": args.log_dir,
                "num_samples": args.num_samples,
            }
        }

    def _build_multi_run_config(self, args: argparse.Namespace) -> Dict:
        with open(args.run_config) as rc:
            run_config = json.load(rc)

        experiments_config = {}
        for experiment_name, exp_config in run_config.items():
            with open(exp_config['base_config']) as f:
                base_config = f.read()
            run_func = self._runner.get_run_func(base_config=base_config,
                                                 args=args)
            run_name = f"run_{experiment_name}"
            register_trainable(run_name, run_func)

            with open(exp_config['search_config']) as f:
                search_config = self.parse_search_config(json.load(f))

            experiments_config[experiment_name] = {
                'run': run_name,
                "resources_per_trial": {
                    "cpu": args.cpus_per_trial,
                    "gpu": args.gpus_per_trial,
                },
                "config": search_config,
                "local_dir": str(Path(args.log_dir, args.run_name)),
                "num_samples": exp_config['num_samples'],
            }

        return experiments_config

    def run_distributed(
        self,
        experiments_config: Dict,
        args: argparse.Namespace,
    ) -> None:
        
        logger.info(
            f"Init Ray with {args.num_cpus} CPUs "
            + f"and {args.num_gpus} GPUs."
        )
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

        logger.info(f"Run Configuration: {experiments_config}")
        try:
            run_experiments(
                experiments=experiments_config,
                scheduler=None,
                with_server=args.with_server,
                server_port=args.server_port,
            )

        except ray.tune.TuneError as e:
            logger.error(
                f"Error during run of experiment: {e}"
            )

    def run(self, args: argparse.Namespace) -> None:
        if args.search_mode == MULTI_PARAM_SEARCH:
            experiments_config = self._build_multi_run_config(args)
        elif args.search_mode == SINGLE_PARAM_SEARCH:
            experiments_config = self._build_single_run_config(args)
        else:
            raise RuntimeError('Unknown hyperparameter search mode')

        print(f"CWD: {os.getcwd()}")
        setattr(args, "cwd", os.getcwd())  # TODO: is this still used?
        self.run_distributed(experiments_config, args)
