import numpy as np

from ber.rescue import first_numbers, rescue_mask


def test_first_numbers_strip_leading_zeros():
    assert first_numbers(
        ["00115 Cours De La Marne", "N° 10 R DE CHEVERUS", "", "Rue X"]
    ).tolist() == [
        "115",
        "10",
        "",
        "",
    ]


def test_rescue_mask_takes_same_address_pool_heavy_word_additions_only():
    s1_names = ["Association de Girlys", "Ets Classe SARL", "Relief Amicale", "Garage Nord"]
    s1_addr = [
        "7 Rue d'Espagne, Nantes, Pays de la Loire",
        "198 Avenue d'Eysines, Bordeaux",
        "15 Boulevard de Verdun, Saint-Nazaire",
        "3 Rue du Port, Calais",
    ]
    pool_names = [
        "Association de Girlys Développement",  # same number, adds a pool-heavy word: yes
        "Ets Classe SARL Développement",  # same number (another format): yes
        "Relief Amicale Holding",  # a planted lookalike: the house number moved
        "Association de Girlys",  # nothing added
        "Garage Nord Sud",  # the added word is not pool-heavy
    ]
    pool_addr = [
        "7 RUE D'ESPAGNE, NANTES",
        "Gironde, 00198 Avenue D'eysines, Bordeaux",
        "19 Bd. De Verdun, St-nazaire",
        "7 Rue d'Espagne",
        "3 R du Port, Calais",
    ]
    rows = np.array([0, 1, 2, 0, 3])
    cols = np.array([0, 1, 2, 3, 4])
    heavy = {"france": {"developpement", "holding"}}
    got = rescue_mask(
        rows, cols, s1_names, s1_addr, ["France"] * 4, pool_names, pool_addr, ["France"] * 5, heavy
    )
    assert got.tolist() == [True, True, False, False, False]


def test_namesake_mask_takes_address_less_exact_names_shared_by_several_businesses():
    from ber.rescue import namesake_mask

    s1_names = ["Club de Nature", "Club de Nature SAS", "Maison Unique", "Club de Nature"]
    s1_country = ["France", "France", "France", "US"]
    pool_names = ["CLUB DE NATURE", "Maison Unique", "Club de Nature Groupe"]
    pool_addr = ["", "", ""]
    rows = np.array([0, 1, 2, 0, 3])
    cols = np.array([0, 0, 1, 2, 0])
    got = namesake_mask(rows, cols, s1_names, s1_country, pool_names, pool_addr)
    # two French namesakes (the legal form does not count), a unique name, a name variant,
    # and the US namesake is alone in its country
    assert got.tolist() == [True, True, False, False, False]


def test_dropped_pairs_stay_unmatchable_through_shift_and_adjustments():
    from ber.config import load_config
    from ber.scale import apply_adjustments, shift_countries

    probs = np.array([0.0, 0.3, 0.0, 0.6], dtype=np.float32)  # 0: dropped by the cascade
    rows = np.array([0, 0, 1, 1])
    shifted = shift_countries(probs, rows, np.array(["france", "france"]), "france:-1.0")
    assert shifted[0] == 0 and shifted[2] == 0 and 0 < shifted[1] < 0.3
    cfg = load_config(None, ["decision.rescue_floor=0.8"])
    rescue = np.array([True, True, False, False])
    delta = np.array([0, 0, -3, -3], dtype=np.float32)
    out = apply_adjustments(shifted, rescue, delta, cfg)
    assert out[0] == 0 and out[2] == 0  # the floor and the shift leave dropped pairs alone
    assert out[1] == np.float32(0.8) and out[3] < shifted[3]


def test_rescue_floor_clears_the_rule_threshold():
    from ber.config import load_config
    from ber.scale import apply_adjustments
    from ber.tuning import Rule

    cfg = load_config(None, ["decision.rescue_floor=0.8"])
    probs = np.array([0.3, 0.3], dtype=np.float32)
    rescue = np.array([True, False])
    # a density-tuned t_rest of 0.82 lifts the floor to 0.87, so a rescued pair still passes
    out = apply_adjustments(
        probs, rescue, None, cfg, Rule("two_threshold", {"t_top": 0.48, "t_rest": 0.82})
    )
    assert np.isclose(out[0], 0.87) and out[1] == np.float32(0.3)
    # a rule below the configured floor leaves it at 0.8
    out = apply_adjustments(
        probs, rescue, None, cfg, Rule("two_threshold", {"t_top": 0.48, "t_rest": 0.74})
    )
    assert out[0] == np.float32(0.8)


def test_replace_mask_finds_one_word_swaps_for_the_listed_words_at_the_same_number():
    from ber.rescue import replace_mask

    s1_names = ["Roubaix Federation EURL", "Lille Club Sportif", "Calais Peinture SAS"]
    s1_addr = ["56 Rue Saint-Hubert, Roubaix", "3 Rue de Lens, Lille", "26 Rue Kellermann"]
    pool_names = [
        "Roubaix EURL & Associés",  # swap to a listed word, same number: kept
        "Roubaix EURL & Associés",  # same name at another number: not kept
        "Lille Club Sportif Associés",  # adds without dropping: the rescue's pattern, not this
        "Calais SAS Bordeaux",  # swap to an unlisted word: not kept
    ]
    pool_addr = [
        "56 R. Saint-hdbert, Roubaix",
        "57 Rue Saint-Hubert",
        "3 Rue de Lens",
        "26 Rue Kellermann",
    ]
    rows = np.array([0, 0, 1, 2])
    cols = np.array([0, 1, 2, 3])
    words = {"france": {"associes", "fils", "groupe"}}
    got = replace_mask(rows, cols, s1_names, s1_addr, ["France"] * 3, pool_names, pool_addr, words)
    assert got.tolist() == [True, False, False, False]


def test_swap_decoy_mask_finds_in_place_swaps_to_common_words():
    from ber.rescue import swap_decoy_mask

    common = ["Lille Ecole SARL", "Roubaix Ecole SAS", "Nantes Ecole", "Calais Club"]
    s1_names = ["Forme Amicale SA", "Motor Tech SARL", "Garde Chasse", *common * 13]
    s1_addr = ["20 Rue du Luxembourg", "11 Rue de la Harpe", "5 Rue X", *["1 Rue Y"] * 52]
    pool_names = [
        "Forme Ecole SA",  # in-place swap to a common word (ecole): capped
        "Motor Centre",  # swap to a kept generic word at the end: not capped
        "Garde Ecole",  # the common word is the last word: not capped
        "Forme Ecole SA",  # another house number: not capped
    ]
    pool_addr = ["20 R Du Luxembourg", "11 Bis Rue De La Harpe", "5 R X", "21 Rue du Luxembourg"]
    rows = np.array([0, 1, 2, 0])
    cols = np.array([0, 1, 2, 3])
    got = swap_decoy_mask(
        rows, cols, s1_names, s1_addr, ["France"] * len(s1_names), pool_names, pool_addr,
        {"france"}, {"centre", "service"}, min_vocab=10,
    )  # fmt: skip
    assert got.tolist() == [True, False, False, False]
