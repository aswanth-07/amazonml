"""Candidate generation: multi-view inverted-index search (``ber.blocking.multiview``).

The search is the first stage of the blocking cascade. ``ber.scale`` adds the embedding
view, re-scores and selects per Source 1 entity, then prunes with stage 1 and the
cross-encoders; the pairs that survive are what the matching model scores and what
``candidate_pairs.tsv`` reports, so final matches are always a subset of them.
"""
