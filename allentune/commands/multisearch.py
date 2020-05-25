#!/usr/bin/env python
import sys
import logging
import os
import argparse

from allentune import MULTI_PARAM_SEARCH
from allentune.modules import AllenNlpRunner
from allentune.modules import RayExecutor
from allentune.commands.subcommand import Subcommand

if os.environ.get("ALLENTUNE_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


class MultiSearch(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
            name,
            description="run multiple experiments with RayTune",
            help='Perform multiple hyperparameter search runs'
        )

        subparser.add_argument(
            "--run-name",
            type=str,
            required=True,
            help="a name for the scheduled runs",
        )
        subparser.add_argument(
            "--num-cpus",
            type=int,
            default=1,
            help="number of CPUs available to the experiment",
        )
        subparser.add_argument(
            "--num-gpus",
            type=int,
            default=1,
            help="number of GPUs available to the experiment",
        )
        subparser.add_argument(
            "--cpus-per-trial",
            type=int,
            default=1,
            help="number of CPUs dedicated to a single trial",
        )
        subparser.add_argument(
            "--gpus-per-trial",
            type=int,
            default=1,
            help="number of GPUs dedicated to a single trial",
        )
        subparser.add_argument(
            "--log-dir",
            type=str,
            default="./logs",
            help="directory in which to store trial logs and results",
        )
        subparser.add_argument(
            "--with-server",
            action="store_true",
            default=False,
            help="start the Ray server",
        )
        subparser.add_argument(
            "--server-port",
            type=int,
            default=10000,
            help="port for Ray server to listens on",
        )
        subparser.add_argument(
            "--search-strategy",
            type=str,
            default="variant-generation",
            help="hyperparameter search strategy used by Ray-Tune",
        )
        subparser.add_argument(
            "--run-config",
            "-c",
            type=os.path.abspath,
            required=True,
            help="name of dict describing the hyperparameter search space",
        )

        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.set_defaults(func=search_from_args)
        subparser.set_defaults(search_mode=MULTI_PARAM_SEARCH)

        return subparser


def search_from_args(args: argparse.Namespace):
    runner = AllenNlpRunner()
    executor = RayExecutor(runner)
    executor.run(args)
