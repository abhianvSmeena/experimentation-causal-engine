import pytest

from experiment_toolkit import design


def test_sample_size_proportions_matches_textbook_value():
    # 10% -> 11% at alpha=0.05, power=0.8: (1.96 + 0.8416)^2 * (0.09 + 0.0979) / 0.01^2
    n_c, n_t = design.sample_size_proportions(0.10, 0.10)
    assert n_c == n_t
    assert n_c == pytest.approx(14_749, abs=2)


def test_sample_size_and_power_are_inverse():
    n_c, n_t = design.sample_size_means(sd=10, mde=0.5)
    assert design.power(n_c, n_t, sd_control=10, effect=0.5) == pytest.approx(0.8, abs=0.005)
    assert design.mde(n_c, n_t, sd_control=10) == pytest.approx(0.5, rel=0.005)


def test_unequal_split_needs_more_users_in_total():
    equal = sum(design.sample_size_means(sd=1, mde=0.1, ratio=1.0))
    skewed = sum(design.sample_size_means(sd=1, mde=0.1, ratio=4.0))
    assert skewed > equal


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        design.sample_size_means(sd=1, mde=0)
    with pytest.raises(ValueError):
        design.sample_size_proportions(0.9, 0.2)
