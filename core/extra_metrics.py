try:
    from federatedscope.register import register_metric
    _HAS_FEDERATEDSCOPE = True
except ImportError:
    # FederatedScope is an optional dependency and is not vendored in this
    # release. This module is a FederatedScope metric plugin: when the framework
    # is absent there is nothing to register into, so fall back to a no-op
    # `register_metric` and keep the metric/dispatch functions importable.
    _HAS_FEDERATEDSCOPE = False

    def register_metric(name, func):  # no-op fallback
        return None

# Define the metric evaluation functions.
# Each reads a single pre-computed score off the FederatedScope context (ctx).
def eval_webshop_task_score(ctx, **kwargs):
    return ctx.webshop_task_score

def eval_test_score(ctx, **kwargs):
    return ctx.test_score

def eval_success_rate(ctx, **kwargs):
    return ctx.success_rate

# extra for alfworld
def pick_two_obj_and_place_success_rate(ctx, **kwargs):
    return ctx.pick_two_obj_and_place_success_rate

def look_at_obj_in_light_success_rate(ctx, **kwargs):
    return ctx.look_at_obj_in_light_success_rate

def pick_and_place_success_rate(ctx, **kwargs):
    return ctx.pick_and_place_success_rate

def pick_clean_then_place_in_recep_success_rate(ctx, **kwargs):
    return ctx.pick_clean_then_place_in_recep_success_rate

def pick_heat_then_place_in_recep_success_rate(ctx, **kwargs):
    return ctx.pick_heat_then_place_in_recep_success_rate

def pick_cool_then_place_in_recep_success_rate(ctx, **kwargs):
    return ctx.pick_cool_then_place_in_recep_success_rate

# Define the dispatch (call) functions. FederatedScope invokes each with the
# requested metric `types`; a matching function returns a 3-tuple of
# (metric_name, eval_function, the_larger_the_better_flag). The third element
# (True here) tells FederatedScope that a higher value is better.
def call_webshop_task_score(types):
    if "webshop_task_score" in types:
        return "webshop_task_score", eval_webshop_task_score, True  # third arg: higher-is-better

def call_test_score(types):
    if "tes_score/text" in types:
        return "tes_score/text", eval_test_score, True  # third arg: higher-is-better

def call_success_rate(types):
    if "success_rate" in types:
        return "success_rate", eval_success_rate, True  # third arg: higher-is-better

# extra for alfworld - every entry passes the higher-is-better flag as the third arg
def call_two_obj_and_place_success_rate(types):
    if "pick_two_obj_and_place_success_rate" in types:
        return "pick_two_obj_and_place_success_rate", pick_two_obj_and_place_success_rate, True

def call_look_at_obj_in_light_success_rate(types):
    if "look_at_obj_in_light_success_rate" in types:
        return "look_at_obj_in_light_success_rate", look_at_obj_in_light_success_rate, True

def call_pick_and_place_success_rate(types):
    if "pick_and_place_success_rate" in types:
        return "pick_and_place_success_rate", pick_and_place_success_rate, True

def call_pick_clean_then_place_in_recep_success_rate(types):
    if "pick_clean_then_place_in_recep_success_rate" in types:
        return "pick_clean_then_place_in_recep_success_rate", pick_clean_then_place_in_recep_success_rate, True

def call_pick_heat_then_place_in_recep_success_rate(types):
    if "pick_heat_then_place_in_recep_success_rate" in types:
        return "pick_heat_then_place_in_recep_success_rate", pick_heat_then_place_in_recep_success_rate, True

def call_pick_cool_then_place_in_recep_success_rate(types):
    if "pick_cool_then_place_in_recep_success_rate" in types:
        return "pick_cool_then_place_in_recep_success_rate", pick_cool_then_place_in_recep_success_rate, True

# Register every metric with the FederatedScope metric registry.
register_metric("webshop_task_score", call_webshop_task_score)
register_metric("tes_score/text", call_test_score)
register_metric("success_rate", call_success_rate)

# extra for alfworld
register_metric("pick_two_obj_and_place_success_rate", call_two_obj_and_place_success_rate)
register_metric("look_at_obj_in_light_success_rate", call_look_at_obj_in_light_success_rate)
register_metric("pick_and_place_success_rate", call_pick_and_place_success_rate)
register_metric("pick_clean_then_place_in_recep_success_rate", call_pick_clean_then_place_in_recep_success_rate)
register_metric("pick_heat_then_place_in_recep_success_rate", call_pick_heat_then_place_in_recep_success_rate)
register_metric("pick_cool_then_place_in_recep_success_rate", call_pick_cool_then_place_in_recep_success_rate)

if _HAS_FEDERATEDSCOPE:
    print("Custom metrics registered successfully!")
else:
    print("FederatedScope not installed; custom metric registration skipped.")
