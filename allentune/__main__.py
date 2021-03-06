#!/usr/bin/env python
import logging
import os
import sys

if os.environ.get("ALLENTUNE_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allentune.commands import main  # pylint: disable=wrong-import-position

def run():
    main(prog="allentune")

if __name__ == "__main__":
    run()