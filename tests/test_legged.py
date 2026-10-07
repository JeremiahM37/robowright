"""The legged contract (it ships in the package as :mod:`robowright.contract.test_legged`).

Collected here too, so that ``pytest tests`` and the commands in CONTRIBUTING.md run it.
"""

from robowright.contract import test_legged as _contract

# Every test and fixture, the module's own (underscored) autouse fixture included.
globals().update({k: v for k, v in vars(_contract).items() if not k.startswith("__")})
