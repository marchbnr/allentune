from allentune import SINGLE_PARAM_SEARCH, MULTI_PARAM_SEARCH
from allentune.modules import AllenNlpRunner, RayExecutor
import pytest
import argparse
import os
import shutil
import pathlib
class TestExampleRun(object):
    
    def test_run(self):
        runner = AllenNlpRunner()
        executor = RayExecutor(runner)
        args = argparse.Namespace()
        PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
        MODULE_ROOT = PROJECT_ROOT / "allentune"
        TESTS_ROOT = MODULE_ROOT / "tests"
        FIXTURES_ROOT = TESTS_ROOT / "fixtures"
        args.experiment_name = "test"
        args.num_cpus = 1
        args.num_gpus = 0
        args.cpus_per_trial = 1
        args.gpus_per_trial = 0
        args.base_config = FIXTURES_ROOT / "classifier.jsonnet"
        args.search_space = FIXTURES_ROOT / "search_space.json"
        args.log_dir = TESTS_ROOT / "logs"
        args.num_samples = 1
        args.with_server = False
        args.server_port = 1000
        args.search_strategy = "variant-generation"
        args.search_mode = SINGLE_PARAM_SEARCH
        executor.run(args)
        assert os.path.isdir(TESTS_ROOT / "logs")
        shutil.rmtree(TESTS_ROOT / "logs/")

    def test_grid_search(self):
        runner = AllenNlpRunner()
        executor = RayExecutor(runner)
        args = argparse.Namespace()
        PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
        MODULE_ROOT = PROJECT_ROOT / "allentune"
        TESTS_ROOT = MODULE_ROOT / "tests"
        FIXTURES_ROOT = TESTS_ROOT / "fixtures"
        args.experiment_name = "grid-search-test"
        args.num_cpus = 1
        args.num_gpus = 0
        args.cpus_per_trial = 1
        args.gpus_per_trial = 0
        args.base_config = FIXTURES_ROOT / "classifier.jsonnet"
        args.search_space = FIXTURES_ROOT / "grid_search_space.json"
        args.log_dir = TESTS_ROOT / "logs"
        args.num_samples = 1
        args.with_server = False
        args.server_port = 1000
        args.search_strategy = "variant-generation"
        args.search_mode = SINGLE_PARAM_SEARCH
        executor.run(args)
        assert os.path.isdir(TESTS_ROOT / "logs")
        shutil.rmtree(TESTS_ROOT / "logs/")

    def test_multiple_runs(self):
        runner = AllenNlpRunner()
        executor = RayExecutor(runner)
        args = argparse.Namespace()
        PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
        MODULE_ROOT = PROJECT_ROOT / "allentune"
        TESTS_ROOT = MODULE_ROOT / "tests"
        FIXTURES_ROOT = TESTS_ROOT / "fixtures"
        args.run_name = "mult-run-test"
        args.num_cpus = 1
        args.num_gpus = 0
        args.cpus_per_trial = 1
        args.gpus_per_trial = 0
        args.log_dir = TESTS_ROOT / "logs"
        args.run_config = FIXTURES_ROOT / "multi-run-config.json"
        args.with_server = False
        args.server_port = 1000
        args.search_mode = MULTI_PARAM_SEARCH
        executor.run(args)
        assert os.path.isdir(TESTS_ROOT / "logs")
        shutil.rmtree(TESTS_ROOT / "logs/")
