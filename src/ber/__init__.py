"""Business entity resolution across three noisy sources.

Pipeline: normalise -> block (candidate generation) -> pair features -> matcher
-> per-entity decision -> ``matching_results.tsv`` + ``candidate_pairs.tsv``.
"""

__version__ = "0.1.0"
