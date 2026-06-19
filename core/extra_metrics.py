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

# Metric evaluation functions. Each reads one pre-computed scalar off the
# FederatedScope evaluation context (ctx) and returns it to the framework.
# Mapping to the paper's reported metrics:
#   success_rate        -> Success Rate  (primary metric; binary task success on the
#                          unperturbed validation env; for ALFWorld it is further
#                          broken down per task type, see the ALFWorld block below).
#   webshop_task_score  -> Task Score    (WebShop secondary metric; reward in [0,1]).
#   test_score          -> the per-data-source validation reward (np.mean of rewards);
#                          for WebShop this is the same Task Score signal under the
#                          generic verl-agent key 'test_score/<data_source>'.
# Note: these key strings are wired across yaml/python/verl-agent, so treat them as
# fixed runtime keys rather than free-form labels. (KNOWN TYPO, left unfixed: the
# call_test_score() dispatcher below matches the type string 'tes_score/text' -- a
# misspelling of 'test_score' plus a hard-coded 'text' data source -- whereas the
# producer emits 'val/test_score/<data_source>'. Verify before relying on test_score
# in the FederatedScope metrics path; this path is separate from the live
# main_ppo_fed training loop, so it does not affect the paper's reported numbers.)
def eval_webshop_task_score(ctx, **kwargs):
    return ctx.webshop_task_score

def eval_test_score(ctx, **kwargs):
    return ctx.test_score

def eval_success_rate(ctx, **kwargs):
    return ctx.success_rate

# ALFWorld-only metrics: per-task-type Success Rate breakdowns (the paper reports
# ALFWorld Success Rate split by task type, plus an All-task aggregate). Each
# function below reads the success rate for one ALFWorld task category off ctx;
# the verbose names map to the paper's task types as:
#   pick_and_place_*                  -> Pick
#   look_at_obj_in_light_*            -> Look
#   pick_clean_then_place_in_recep_*  -> Clean
#   pick_heat_then_place_in_recep_*   -> Heat
#   pick_cool_then_place_in_recep_*   -> Cool
#   pick_two_obj_and_place_*          -> Pick2
# (The function/key names mirror ctx attributes set by the ALFWorld eval, so they
# are runtime keys and must not be renamed.)
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

# Register the ALFWorld per-task-type Success Rate metrics defined above
# (Pick / Look / Clean / Heat / Cool / Pick2). No-op when FederatedScope is absent.
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
