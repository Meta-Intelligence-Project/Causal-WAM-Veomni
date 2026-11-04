# SPDX-License-Identifier: OpenMDW-1.1

import math
from collections import deque

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy, fit_latency


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_public_fit_reports_unavailable_without_a_completed_or_observed_latency():
    estimate = fit_latency(2.4)
    assert estimate == dict(
        fit_mode="unavailable",
        sample_count=0,
        distinct_history_count=0,
        history_min_seconds=None,
        history_max_seconds=None,
        intercept_seconds=0.0,
        slope_seconds_per_history_second=0.0,
        predicted_seconds=0.0,
        residual_seconds=0.0,
        maximum_seconds=0.0,
    )


def test_public_fit_reports_constant_for_a_fixed_encoded_padded_history():
    history = 1024 / 30
    estimate = fit_latency(history, samples=((history, 1.1), (history, 1.3)))
    assert estimate["fit_mode"] == "constant"
    assert estimate["sample_count"] == 2 and estimate["distinct_history_count"] == 1
    assert estimate["history_min_seconds"] == estimate["history_max_seconds"] == history
    assert estimate["intercept_seconds"] == estimate["predicted_seconds"] == estimate["maximum_seconds"] == 1.3
    assert estimate["slope_seconds_per_history_second"] == estimate["residual_seconds"] == 0.0


def test_public_fit_extrapolates_the_cpu_line_without_execution_policy_margins():
    estimate = fit_latency(6, samples=((0, 1), (2, 2), (4, 3)))
    assert estimate["fit_mode"] == "linear"
    assert estimate["sample_count"] == estimate["distinct_history_count"] == 3
    assert estimate["intercept_seconds"] == pytest.approx(1)
    assert estimate["slope_seconds_per_history_second"] == pytest.approx(0.5)
    assert estimate["predicted_seconds"] == pytest.approx(4)
    assert estimate["maximum_seconds"] == 3
    assert estimate["residual_seconds"] == pytest.approx(0)


@pytest.mark.parametrize("samples", [(), ((0, 1), (2, 2), (4, 3))])
def test_public_fit_does_not_count_elapsed_lower_bounds_as_completed_samples(samples):
    estimate = fit_latency(6, samples=samples, observed_seconds=10)
    assert estimate["sample_count"] == len(samples)
    assert estimate["maximum_seconds"] == (3 if samples else 0)
    assert estimate["predicted_seconds"] == (4 if samples else 10)


@pytest.mark.parametrize(
    "options",
    [
        dict(history_seconds=True),
        dict(history_seconds=-1),
        dict(history_seconds=float("nan")),
        dict(history_seconds=float("inf")),
        dict(observed_seconds=True),
        dict(observed_seconds=-1),
        dict(observed_seconds=float("nan")),
        dict(observed_seconds=float("inf")),
        dict(samples=None),
        dict(samples=((0,),)),
        dict(samples=((-1, 2),)),
        dict(samples=((1, -2),)),
        dict(samples=((True, 2),)),
        dict(samples=((1, float("nan")),)),
    ],
)
def test_public_fit_rejects_invalid_inputs_without_a_policy(options):
    with pytest.raises(ValueError):
        fit_latency(**({"history_seconds": 0} | options))


def recommend(policy, *, history=3.0, rate=3.0, samples=((0, 2), (1, 3), (2, 4)), **options):
    return policy.recommend(history, horizon=32, execution_steps=16, current_hz=rate, samples=samples, **options)


def test_no_latency_is_fabricated_before_startup_measurement():
    advice = recommend(RTCExecutionPolicy(), samples=())
    assert not advice.feasible
    assert advice.reason == "no_latency_samples"
    assert advice.budget_seconds is None
    assert advice.sample_count == 0


def test_cpu_line_is_fitted_from_the_supplied_completed_requests():
    advice = recommend(RTCExecutionPolicy())
    assert advice.feasible and advice.fit_mode == "linear"
    assert advice.sample_count == advice.distinct_history_count == 3
    assert (advice.history_min_seconds, advice.history_max_seconds) == (0, 2)
    assert advice.intercept_seconds == pytest.approx(2)
    assert advice.slope_seconds_per_history_second == pytest.approx(1)
    assert advice.predicted_seconds == pytest.approx(5)
    assert advice.residual_seconds == pytest.approx(0)
    assert advice.maximum_seconds == 4
    assert advice.budget_seconds == pytest.approx(6)
    assert advice.recommended_hz == pytest.approx(2.5)


def test_an_updated_buffer_changes_the_prediction_without_changing_policy_parameters():
    policy = RTCExecutionPolicy()
    before = recommend(policy, samples=((0, 2), (1, 3)))
    after = recommend(policy, samples=((0, 2), (1, 4)))
    assert before.slope_seconds_per_history_second == 1
    assert after.slope_seconds_per_history_second == 2
    assert after.predicted_seconds > before.predicted_seconds
    assert after.recommended_hz < before.recommended_hz


def test_a_single_history_uses_the_largest_measured_latency_until_a_slope_is_observed():
    advice = recommend(RTCExecutionPolicy(), samples=((0, 2), (0, 3)), history=1)
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.distinct_history_count == 1
    assert advice.slope_seconds_per_history_second == 0
    assert advice.predicted_seconds == 3
    assert advice.budget_seconds == pytest.approx(3.6)


def test_a_full_window_with_only_one_history_length_is_not_singular():
    advice = recommend(RTCExecutionPolicy(), history=1024 / 30, samples=((1024 / 30, 12), (1024 / 30, 13)))
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.predicted_seconds == 13


def test_negative_observed_slope_cannot_predict_that_more_history_becomes_faster():
    advice = recommend(RTCExecutionPolicy(), samples=((0, 4), (1, 3), (2, 2)))
    assert advice.slope_seconds_per_history_second == 0
    assert advice.budget_seconds >= 4 * 1.2


def test_positive_residual_is_added_to_the_next_prediction():
    advice = recommend(RTCExecutionPolicy(latency_factor=1), samples=((0, 2), (1, 4), (2, 4)))
    assert advice.predicted_seconds == pytest.approx(16 / 3)
    assert advice.residual_seconds == pytest.approx(2 / 3)
    assert advice.budget_seconds == pytest.approx(6)


def test_a_local_line_cannot_produce_a_negative_latency_budget():
    advice = recommend(RTCExecutionPolicy(), history=0, samples=((10, 1), (11, 2)))
    assert advice.intercept_seconds == pytest.approx(-9)
    assert advice.predicted_seconds == 0
    assert advice.budget_seconds == pytest.approx(2.4)


def test_no_offline_maximum_history_is_required_for_online_prediction():
    advice = recommend(RTCExecutionPolicy(), history=36, samples=((34, 12), (35, 13)))
    assert advice.feasible
    assert advice.predicted_seconds == pytest.approx(14)


def test_only_the_current_bounded_buffer_is_used_without_retained_fit_state():
    samples = deque([(0, 100), (1, 100)], maxlen=2)
    policy = RTCExecutionPolicy()
    first = recommend(policy, samples=samples)
    samples.extend([(2, 3), (3, 4)])
    second = recommend(policy, history=4, samples=samples)
    assert first.predicted_seconds == 100
    assert second.sample_count == 2
    assert second.history_min_seconds == 2
    assert second.predicted_seconds == pytest.approx(5)


def test_large_absolute_histories_preserve_small_differences_when_fitting():
    origin = 1e12
    advice = recommend(
        RTCExecutionPolicy(), history=origin + 3, samples=((origin, 2), (origin + 1, 3), (origin + 2, 4))
    )
    assert advice.slope_seconds_per_history_second == 1
    assert advice.predicted_seconds == 5


def test_recent_wall_latency_and_transport_headroom_override_the_fit():
    advice = recommend(RTCExecutionPolicy(extra_latency_seconds=0.5), observed_seconds=10)
    assert advice.budget_seconds == 12.5
    assert advice.recommended_hz == pytest.approx(15 / 12.5)


def test_observed_seconds_can_supply_a_constant_startup_estimate():
    advice = recommend(RTCExecutionPolicy(), samples=(), observed_seconds=2)
    assert advice.feasible and advice.fit_mode == "constant"
    assert advice.sample_count == 0
    assert advice.budget_seconds == pytest.approx(2.4)


def test_policy_does_not_speed_up_after_a_lower_latency_observation():
    policy = RTCExecutionPolicy()
    first = recommend(policy, observed_seconds=20)
    second = recommend(policy, rate=first.recommended_hz, samples=((0, 1), (1, 1)))
    assert first.feasible and second.feasible
    assert second.recommended_hz == first.recommended_hz


def test_history_growth_reduces_rate_before_the_prefix_budget_is_exhausted():
    policy = RTCExecutionPolicy()
    samples = deque(maxlen=8)
    rate = 5.0
    for block in range(33):
        history = block * 32 / 30
        samples.append((history, 2 + 0.17 * history + 0.003 * history * history))
        advice = recommend(policy, history=history + 32 / 30, rate=rate, samples=samples)
        assert advice.feasible and 0 < advice.recommended_hz <= rate
        assert math.ceil(advice.budget_seconds * advice.recommended_hz) + policy.margin_steps <= 16
        rate = advice.recommended_hz
    assert rate < 1.2


def test_minimum_rate_can_make_slowdown_infeasible():
    advice = recommend(RTCExecutionPolicy(minimum_execution_hz=2), samples=((0, 10),))
    assert not advice.feasible and advice.reason == "minimum_execution_hz"
    assert advice.recommended_hz == 3


def test_exactly_feasible_minimum_rate_is_accepted():
    advice = recommend(RTCExecutionPolicy(latency_factor=1, minimum_execution_hz=2), samples=((0, 7.5),))
    assert advice.feasible and advice.recommended_hz == 2
    assert advice.delay_steps == 16


def test_asymmetric_replanning_uses_the_smaller_available_prefix():
    policy = RTCExecutionPolicy(margin_steps=2)
    advice = policy.recommend(0, horizon=32, execution_steps=24, current_hz=3, samples=((0, 10),))
    assert advice.feasible and advice.delay_steps == 8
    assert advice.recommended_hz == pytest.approx(6 / 12)
    assert not policy.recommend(0, horizon=32, execution_steps=1, current_hz=3, samples=((0, 10),)).feasible


@pytest.mark.parametrize("budget", [0.1, 0.3, 1, 3.14, 11.975, 15.484, 1000])
def test_float_boundaries_preserve_the_integer_deadline_inequality(budget):
    advice = recommend(RTCExecutionPolicy(latency_factor=1), rate=10000, samples=((0, budget),))
    assert advice.feasible
    assert math.ceil(budget * advice.recommended_hz) + 1 <= 16


@pytest.mark.parametrize(
    "options",
    [
        dict(latency_factor=0.9),
        dict(latency_factor=True),
        dict(extra_latency_seconds=float("inf")),
        dict(extra_latency_seconds=-1),
        dict(margin_steps=True),
        dict(margin_steps=0),
        dict(minimum_execution_hz=-1),
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        RTCExecutionPolicy(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(history_seconds=-1),
        dict(horizon=True),
        dict(execution_steps=32),
        dict(current_hz=0),
        dict(observed_seconds=float("nan")),
        dict(samples=None),
        dict(samples=((0,),)),
        dict(samples=((-1, 2),)),
        dict(samples=((1, -2),)),
        dict(samples=((True, 2),)),
        dict(samples=((1, float("nan")),)),
    ],
)
def test_invalid_requests_are_rejected(options):
    arguments = dict(history_seconds=0, horizon=32, execution_steps=16, current_hz=3, samples=((0, 2),))
    with pytest.raises(ValueError):
        RTCExecutionPolicy().recommend(**(arguments | options))


def handoff(policy=None, **options):
    arguments = dict(history_seconds=0, overlap_steps=8, current_hz=30, samples=((0, 2),))
    return (policy or RTCExecutionPolicy()).recommend_handoff(**(arguments | options))


def test_fixed_handoff_uses_only_the_overlap_budget_and_preserves_its_delay():
    advice = handoff()
    assert advice.feasible
    assert advice.budget_seconds == pytest.approx(2.4)
    assert advice.recommended_hz == pytest.approx(7 / 2.4)
    assert advice.delay_steps == 8
    # A small observed delay does not shrink the hard committed prefix.
    fast = handoff(samples=((0, 0.01),))
    assert fast.recommended_hz == 30
    assert fast.delay_steps == 8


def test_fixed_handoff_reuses_the_online_history_fit():
    advice = handoff(history_seconds=3, samples=((0, 2), (1, 3), (2, 4)))
    assert advice.feasible and advice.fit_mode == "linear"
    assert advice.predicted_seconds == pytest.approx(5)
    assert advice.maximum_seconds == 4
    assert advice.budget_seconds == pytest.approx(6)
    assert advice.recommended_hz == pytest.approx(7 / 6)
    assert advice.delay_steps == 8


def test_fixed_handoff_does_not_fabricate_startup_measurements():
    advice = handoff(samples=())
    assert not advice.feasible and advice.reason == "no_latency_samples"
    assert advice.budget_seconds is None
    assert advice.delay_steps == 8


def test_pending_handoff_spends_only_the_remaining_wall_budget():
    advice = handoff(elapsed_seconds=1, remaining_steps=3)
    assert advice.feasible
    assert advice.recommended_hz == pytest.approx(3 / 1.4)
    assert advice.delay_steps == 8


def test_last_old_command_still_has_one_interval_before_handoff():
    advice = handoff(elapsed_seconds=2, remaining_steps=1, current_hz=3)
    assert advice.feasible
    assert advice.recommended_hz == pytest.approx(2.5)
    assert advice.delay_steps == 8


def test_pending_elapsed_time_is_a_lower_bound_on_total_latency():
    advice = handoff(elapsed_seconds=3, remaining_steps=1)
    assert advice.feasible
    assert advice.budget_seconds == pytest.approx(3.6)
    assert advice.recommended_hz == pytest.approx(1 / 0.6)


def test_exhausted_pending_estimate_is_not_reported_as_a_deadline_guarantee():
    advice = handoff(RTCExecutionPolicy(latency_factor=1), elapsed_seconds=2, remaining_steps=1)
    assert not advice.feasible and advice.reason == "latency_estimate_exhausted"
    assert advice.delay_steps == 8
    assert advice.recommended_hz == 30


@pytest.mark.parametrize("margin", [8, 9])
def test_fixed_handoff_reports_a_margin_that_exhausts_the_overlap(margin):
    advice = handoff(RTCExecutionPolicy(margin_steps=margin))
    assert not advice.feasible and advice.reason == "margin_exhausts_prefix"
    assert advice.delay_steps == 8


def test_no_rate_can_create_more_control_steps_at_the_handoff():
    advice = handoff(elapsed_seconds=2, remaining_steps=0)
    assert not advice.feasible and advice.reason == "no_remaining_steps"
    assert advice.delay_steps == 8


@pytest.mark.parametrize("pending", [False, True])
def test_fixed_handoff_minimum_rate_is_enforced(pending):
    options = dict(elapsed_seconds=1, remaining_steps=1) if pending else {}
    advice = handoff(RTCExecutionPolicy(minimum_execution_hz=10), **options)
    assert not advice.feasible and advice.reason == "minimum_execution_hz"
    assert advice.delay_steps == 8


def test_fixed_handoff_accepts_an_exactly_feasible_minimum_rate():
    advice = handoff(RTCExecutionPolicy(latency_factor=1, minimum_execution_hz=2), samples=((0, 3.5),))
    assert advice.feasible and advice.recommended_hz == 2
    assert advice.delay_steps == 8


def test_fixed_handoff_only_slows_and_never_accelerates_execution():
    before = handoff()
    after = handoff(current_hz=before.recommended_hz, samples=((0, 0.01),))
    assert after.feasible and after.recommended_hz == before.recommended_hz
    pending = handoff(current_hz=0.5, elapsed_seconds=1, remaining_steps=2)
    assert pending.feasible and pending.recommended_hz == 0.5


@pytest.mark.parametrize("duration", [0.1, 0.3, 1, 3.14, 11.975, 15.484, 1000])
@pytest.mark.parametrize("pending", [False, True])
def test_fixed_handoff_float_boundaries_preserve_the_wall_deadline(duration, pending):
    options = dict(elapsed_seconds=duration / 2, remaining_steps=3) if pending else {}
    advice = handoff(RTCExecutionPolicy(latency_factor=1), current_hz=10000, samples=((0, duration),), **options)
    assert advice.feasible
    capacity = 3 if pending else 7
    remaining = duration / 2 if pending else duration
    assert remaining * advice.recommended_hz <= capacity
    assert advice.delay_steps == 8


@pytest.mark.parametrize(
    "options",
    [
        dict(overlap_steps=True),
        dict(overlap_steps=0),
        dict(overlap_steps=8.5),
        dict(remaining_steps=True),
        dict(remaining_steps=-1),
        dict(remaining_steps=9),
        dict(remaining_steps=1.5),
        dict(elapsed_seconds=-1),
        dict(elapsed_seconds=float("inf")),
        dict(elapsed_seconds=1),
        dict(observed_seconds=float("nan")),
        dict(current_hz=0),
        dict(history_seconds=-1),
        dict(samples=None),
    ],
)
def test_invalid_fixed_handoff_requests_are_rejected(options):
    with pytest.raises(ValueError):
        handoff(**options)
