import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Any, Dict, List, Optional

_CLIENT: Optional[OpenAI] = None


def _short_text(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text[:limit]


def short_completion_reward_func(prompts, completions, **kwargs):
    rewards = []
    target_len = int(os.getenv("SHORT_REWARD_TARGET_LEN", "120"))
    min_reward = float(os.getenv("SHORT_REWARD_MIN", "-0.5"))

    for idx, cand in enumerate(completions):
        text = "" if cand is None else str(cand).strip()
        length = len(text)

        if length <= target_len:
            score = 1.0
        else:
            overflow = min(length - target_len, target_len)
            score = 1.0 - (overflow / target_len) * 1.5
            score = max(min_reward, score)

        rewards.append(float(score))
        print(
            f"[short_completion_reward_func] idx={idx} len={length} reward={score:.3f}",
            flush=True,
        )

    return rewards

def traffic_reward_func(prompts, completions, gt=None, assistant=None, **kwargs):
    # TRL may pass a list of strings or a single string depending on batching
    if gt is None:
        gt = assistant  # <-- use dataset's assistant as GT
    if gt is None:
        return [0.0] * len(completions)

    # normalize gt to a list aligned with completions
    if isinstance(gt, str):
        gt = [gt] * len(completions)

    rewards = []
    if len(gt) != len(completions):
        print(
            f"[traffic_reward_func] length mismatch: len(completions)={len(completions)} len(gt)={len(gt)}",
            flush=True,
        )

    num_items = len(completions)
    if num_items == 0:
        return rewards

    max_workers_env = os.getenv("TRAFFIC_REWARD_MAX_WORKERS", "6")
    try:
        max_workers = int(max_workers_env)
    except ValueError:
        max_workers = 6
    max_workers = max(1, min(max_workers, num_items))

    def _score_one(idx: int, cand: Any, gt_i: Any):
        results = g_eval(GPT_MODEL_ID, cand, gt_i)
        score = float(results["final_score"])
        return idx, cand, gt_i, score, results

    rewards = [0.0] * num_items

    if max_workers == 1:
        for idx, cand in enumerate(completions):
            gt_i = gt[idx] if idx < len(gt) else None
            try:
                _, cand, gt_i, score, results = _score_one(idx, cand, gt_i)
                rewards[idx] = score
                print(
                    f"[traffic_reward_func] g_eval results: {results} "
                    f"| cand[:80]={_short_text(cand)!r} | gt[:80]={_short_text(gt_i)!r}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[traffic_reward_func] g_eval failed: {repr(e)} "
                    f"| cand[:80]={_short_text(cand)!r} | gt[:80]={_short_text(gt_i)!r}",
                    flush=True,
                )
                rewards[idx] = 0.0
        return rewards

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_payload = {}
        for idx, cand in enumerate(completions):
            gt_i = gt[idx] if idx < len(gt) else None
            future = executor.submit(_score_one, idx, cand, gt_i)
            future_to_payload[future] = (idx, cand, gt_i)

        for future in as_completed(future_to_payload):
            idx, cand, gt_i = future_to_payload[future]
            try:
                idx, cand, gt_i, score, results = future.result()
                rewards[idx] = score
                print(
                    f"[traffic_reward_func] g_eval results: {results} "
                    f"| cand[:80]={_short_text(cand)!r} | gt[:80]={_short_text(gt_i)!r}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[traffic_reward_func] g_eval failed: {repr(e)} "
                    f"| cand[:80]={_short_text(cand)!r} | gt[:80]={_short_text(gt_i)!r}",
                    flush=True,
                )
                rewards[idx] = 0.0
    return rewards


# ── G-Eval reward (Summariser) ────────────────────────────────────────────────
# GPT_EVAL system prompt and user prompt templates for reward function
GPT_MODEL_ID = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
SYS_PROMPT = """
You are an automatic evaluator used as a reward model for post training. Score a machine generated traffic anomaly summary (CAND) against a human labeled ground truth (GT).

Input
You will receive exactly two texts:
1) GT
2) CAND

Goal
Produce a stable and fair reward signal aligned with the main evaluation metric, while discouraging hallucination and unnecessary verbosity.

General rules
1) Judge CAND only relative to GT. Do not use outside knowledge.
2) Do not reward details unless GT explicitly states them or clearly implies them.
3) Contradictions are incorrect.
4) If GT is silent about a factor, exclude it from scoring for that factor.
5) Use meaning based matching, not exact string matching, where equivalence rules are provided.
6) If GT provides multiple acceptable options, CAND is correct if it matches at least one.
7) If CAND provides multiple options for a factor, mark it correct only if at least one matches GT and none directly contradict GT.
8) Prefer conservative scoring when ambiguous.
9) Return only the JSON object at the end. No extra text.

Scores
Positive subscores:
A) environment_score in [0, 1]
B) grounding_score in [0, 2]
C) description_accuracy_score in [0, 5]
D) reasoning_score in [0, 2]

Penalties:
E) hallucination_penalty in [-3, 0]
F) verbosity_penalty in [-1, 0]

positive_total_score = A + B + C + D
final_score = positive_total_score + E + F

Gating rules
G1) If CAND core event type differs from GT, then description_accuracy_score <= 1.0.
G2) If description_accuracy_score <= 1.0, then reasoning_score <= 1.0.
G3) Any severe hallucination is a major contradiction for description scoring.
G4) If a severe hallucination changes anomaly status, core event type, anomalous agent, collision status, or pedestrian involvement, then description_accuracy_score <= 0.5 and reasoning_score = 0.0.

Severe hallucination definition
Severe means an unsupported claim that changes the core event. Treat as severe if it changes any of:
- anomaly vs normal status
- core event type
- primary anomalous agent
- change the meaning of key details like collision vs no collision, pedestrian involvement, etc.

Do not count as severe: harmless paraphrases, generic safety commentary, minor extra wording that does not change the event.

A) Environment Correctness (0 to 1)
Factors and weights:
time_of_day 1, weather 1, road_surface 1, road_type 3

Canonical labels:
time_of_day: day, night, dawn_dusk, unknown
weather: clear, cloudy, rainy, snowy, foggy, unknown
road_surface: dry, wet, snow_covered, unknown
road_type: single_lane_roundabout, double_lane_roundabout, multi_lane_roundabout, unknown

Procedure:
1) Identify which factors GT specifies (explicit or clearly implied).
2) Score only specified factors. Mark correct if CAND matches meaning.
3) environment_score = weighted_correct / weighted_specified.
4) If weighted_specified = 0, set environment_score = 1.0.

B) Object Grounding (0 to 2)
Grounding measures anomalous agent identity and location.

B1) Identity (0 to 1)
Type list:
sedan, suv, pickup, van, minivan, bus, truck, hgv, motorcycle, bicycle, pedestrian, emergency_vehicle, trailer_towed_object, car, unknown
Color list:
black, white, red, blue, yellow, orange, silver_gray, light_colored, dark_colored, unknown

Scoring:
- type match worth 0.5 if GT specifies type
- color match worth 0.5 if GT specifies color
- exclude missing GT subfields from denominator
Type rules:
- if GT is specific, CAND saying only car is not a match
- if GT says car, sedan or suv or pickup or van or minivan are compatible
- if GT says unknown, only unknown matches
Color rules:
- GT light_colored accepts white or silver_gray
- GT dark_colored accepts black
- GT specific color does not accept light_colored or dark_colored
- if GT says unknown, only unknown matches

Normalization:
earned_B1 is sum earned parts, max_B1 is sum scorable parts.
If max_B1 = 0, set B1_points = 0.5. Else B1_points = earned_B1 / max_B1.

B2) Location (0 to 1)
Frame labels:
upper_left, upper_center, upper_right, center_left, center, center_right, lower_left, lower_center, lower_right, unknown
Accept simple synonyms and formatting variants if meaning is unchanged.

Environment labels:
inner_entry_lane, outer_entry_lane, enter_lane,
inner_exit_lane, outer_exit_lane, exit_lane,
yielding_lane,
inner_circulating_lane, outer_circulating_lane, circulating_lane,
crosswalk_area, central_island, grass_verge, sidewalk, unknown

Environment matching:
- if GT is coarse, specific child labels can match if not contradictory
  enter_lane accepts inner_entry_lane or outer_entry_lane
  exit_lane accepts inner_exit_lane or outer_exit_lane
  circulating_lane accepts inner_circulating_lane or outer_circulating_lane
- if GT is specific, incompatible specific label is incorrect

Phases:
GT may specify start_location, anomaly_location, exit_location.
For frame and environment separately:
- split that field max (0.5) equally across phases GT specifies for that field
- award phase credit if CAND matches
- if GT gives multiple acceptable locations, match any
- if CAND lists multiple, correct only if at least one matches and none contradict

Normalization:
earned_B2 is sum earned phase credits, max_B2 is sum scorable phase credits.
If max_B2 = 0, set B2_points = 0.5. Else B2_points = earned_B2 / max_B2.

grounding_score = B1_points + B2_points.

C) Event Description Accuracy (0 to 5)
Main score: does CAND describe the same anomaly event as GT.

Rubric:
5: Same event type and fully consistent, correct participants, action, outcome, no major contradictions
4: Same event type, mostly consistent, only minor errors or omissions
3: Same event type, misses multiple key details or has one major unsupported detail that does not change the core event
2: Some overlap but key action, participants, or outcome are wrong, or multiple major unsupported details
1: Related setting but core anomaly wrong, or event type mismatch with weak overlap
0: Fundamentally different or contradicts GT

Focus on: anomaly vs normal, core event type, main action, anomalous agent, outcome.
Apply G1, G3, G4.

D) Anomaly Reasoning (0 to 2)
Reasoning explains why it is anomalous, risky, causal, or rule violating.

If GT provides no reasoning:
2: no reasoning or minimal restatement without new risky claims
1: plausible and consistent, no risky unsupported specifics
0: contradicts GT or relies on risky unsupported specifics

If GT provides reasoning:
2: matches or fully consistent
1: different but reasonable and consistent with GT and description
0: contradicts GT or contradicts CAND description

Apply G2.

E) Hallucination penalty (-3 to 0)
0.0: no meaningful hallucination
-0.5: one minor unsupported detail, core event still correct
-1.0: multiple minor unsupported details
-2.0: one severe hallucination
-3.0: multiple severe hallucinations, or one severe hallucination that changes the whole event

F) Verbosity penalty (-1 to 0)
0.0: concise and focused
-0.25: somewhat wordy
-0.5: clearly too long or repetitive
-1.0: very long and redundant

Output JSON
Return only one JSON object:

{
  "environment_score": number,
  "grounding_score": number,
  "description_accuracy_score": number,
  "reasoning_score": number,
  "positive_total_score": number,
  "hallucination_penalty": number,
  "verbosity_penalty": number,
  "final_score": number
}

Constraints:
environment_score in [0, 1]
grounding_score in [0, 2]
description_accuracy_score in [0, 5]
reasoning_score in [0, 2]
positive_total_score = environment_score + grounding_score + description_accuracy_score + reasoning_score
hallucination_penalty in [-3, 0]
verbosity_penalty in [-1, 0]
final_score = positive_total_score + hallucination_penalty + verbosity_penalty

Round all numeric fields to 4 decimals.
""".strip()

def build_message(pred: str, gt: str) -> List[Dict[str, str]]:
    user_prompt = (
        "GROUND TRUTH (GT):\n"
        f"{gt}\n\n"
        "CANDIDATE (CAND):\n"
        f"{pred}\n"
    )
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def g_eval(model: str, pred: str, gt: str) -> Dict[str, Any]:
    client = get_client()
    messages = build_message(pred, gt)

    def _call(msgs):
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            response_format={"type": "json_object"},
        )
        return (r.choices[0].message.content or "").strip()

    def _parse_json(s: str) -> Dict[str, Any]:
        return json.loads(s)

    # Attempt 1
    text = _call(messages)
    try:
        out = _parse_json(text)
    except Exception:
        # Attempt 2: repair prompt
        repair_messages = messages + [{
            "role": "user",
            "content": "Your previous output was not valid JSON. Return ONLY the JSON object, nothing else."
        }]
        text2 = _call(repair_messages)
        out = _parse_json(text2)

    # Basic validation / clamping
    # ensure numeric fields exist
    for k in ["environment_score","grounding_score","description_accuracy_score","reasoning_score","verbosity_penalty","final_score"]:
        if k not in out:
            raise ValueError(f"Missing key {k} in judge output: {out}")

    # clamp final score to its theoretical range [-1, 10]
    out["final_score"] = float(out["final_score"])
    out["final_score"] = max(-1.0, min(10.0, out["final_score"]))
    return out

# ── Category reward (Classifier) ─────────────────────────────────────────────
def category_reward_func(prompts, completions, gt=None, assistant=None, **kwargs):
    # Reward hyperparams
    P_FP = -0.5      # predicted abnormal when GT normal
    P_FN = -0.75     # predicted normal when GT abnormal
    R_CAT_OK = 1.5   # correct A/B/C/D label
    P_BAD_FMT = -2.0 # unparseable / invalid output
    P_SUB = -0.75    # wrong abnormal category (B vs C vs D)

    # Use assistant as fallback GT when gt is not provided.
    if gt is None:
        gt = assistant
    if gt is None:
        return [0.0] * len(completions)

    if isinstance(gt, str):
        gt = [gt] * len(completions)

    valid_labels = {"A", "B", "C", "D"}
    abnormal_labels = {"B", "C", "D"}

    def _normalize_label(value):
        if value is None:
            return None

        if isinstance(value, list) and value:
            if isinstance(value[-1], dict) and "content" in value[-1]:
                value = value[-1]["content"]
            else:
                value = value[-1]

        if isinstance(value, dict):
            value = value.get("content", value.get("text", ""))

        text = str(value).strip().upper()
        if text in valid_labels:
            return text

    rewards = []
    for idx, cand in enumerate(completions):
        gt_i = gt[idx] if idx < len(gt) else None

        if cand is None and gt_i is None:
            rewards.append(0.0)
            print(
                f"[category_reward_func] idx={idx} pred=None gt=None reward=0.00 reason=both_none",
                flush=True,
            )
            continue

        pred_label = _normalize_label(cand)
        gt_label = _normalize_label(gt_i)

        if pred_label is None or gt_label is None:
            rewards.append(P_BAD_FMT)
            print(
                f"[category_reward_func] idx={idx} pred={pred_label} gt={gt_label} "
                f"reward={P_BAD_FMT:.2f} reason=bad_format cand={str(cand)[:60]!r} gt_raw={str(gt_i)[:60]!r}",
                flush=True,
            )
            continue

        pred_is_abnormal = pred_label in abnormal_labels
        gt_is_abnormal = gt_label in abnormal_labels

        if pred_is_abnormal and (not gt_is_abnormal):
            score = P_FP
        elif (not pred_is_abnormal) and gt_is_abnormal:
            score = P_FN
        else:
            score = 0.0

        if pred_label == gt_label:
            score += R_CAT_OK
        else:
            score += P_SUB

        rewards.append(score)
        print(
            f"[category_reward_func] idx={idx} pred={pred_label} gt={gt_label} reward={score:.2f}",
            flush=True,
        )

    return rewards