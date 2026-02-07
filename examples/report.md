# PromptBench Report

**Run ID:** run_1770502148_09d090dd
**Model:** mock
**Schema Version:** 1.0

## Summary

| Scenario | Family | Seeds | Task Success (median) | Efficiency (median) |
|----------|--------|-------|-----------------------|---------------------|
| classification/sentiment | classification | 3 | 0.000 | 0.985 |
| constraint/json_schema | constraint | 3 | 1.000 | 0.967 |
| convergence/error_handling | convergence | 3 | 1.000 | 0.873 |
| tool_use/research_calculate | tool_use | 3 | 0.000 | 0.867 |

## Family: CLASSIFICATION

### Classification - Sentiment
*classification/sentiment* | Seeds: 3

| Metric | Median | Mean | Std | P10 | P90 |
|--------|--------|------|-----|-----|-----|
| accuracy | 0.3636 | 0.3939 | 0.1134 | 0.2909 | 0.5091 |
| consistency | 0.6667 | 0.5556 | 0.1571 | 0.4000 | 0.6667 |
| efficiency | 0.9853 | 0.9853 | 0.0000 | 0.9853 | 0.9853 |
| predictions_made | 11.0000 | 11.0000 | 0.0000 | 11.0000 | 11.0000 |
| task_success | 0.0000 | 0.3333 | 0.4714 | 0.0000 | 0.8000 |

## Family: CONSTRAINT

### Constraint - Json Schema
*constraint/json_schema* | Seeds: 3

| Metric | Median | Mean | Std | P10 | P90 |
|--------|--------|------|-----|-----|-----|
| attempts | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| efficiency | 0.9670 | 0.9670 | 0.0000 | 0.9670 | 0.9670 |
| first_attempt_pass | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| pass_rate | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| recovery_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| task_success | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Family: CONVERGENCE

### Convergence - Error Handling
*convergence/error_handling* | Seeds: 3

| Metric | Median | Mean | Std | P10 | P90 |
|--------|--------|------|-----|-----|-----|
| bonus_coverage | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| convergence_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| efficiency | 0.8732 | 0.8732 | 0.0000 | 0.8732 | 0.8732 |
| invariant_coverage | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| invariants_matched | 5.0000 | 5.0000 | 0.0000 | 5.0000 | 5.0000 |
| invariants_total | 5.0000 | 5.0000 | 0.0000 | 5.0000 | 5.0000 |
| keyword_coverage | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| task_success | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## Family: TOOL_USE

### Tool Use - Research Calculate
*tool_use/research_calculate* | Seeds: 3

| Metric | Median | Mean | Std | P10 | P90 |
|--------|--------|------|-----|-----|-----|
| answer_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| efficiency | 0.8667 | 0.8667 | 0.0000 | 0.8667 | 0.8667 |
| recovery_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| task_success | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tool_call_correctness | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| tools_used | 3.0000 | 3.0000 | 0.0000 | 3.0000 | 3.0000 |
| total_tool_calls | 3.0000 | 3.0000 | 0.0000 | 3.0000 | 3.0000 |

## Pareto Ranking

| Rank | Scenario | Pareto Optimal | Objectives |
|------|----------|----------------|------------|
| 1 | classification/sentiment | Yes | task_success: 0.000, efficiency: 0.985 |
| 1 | constraint/json_schema | Yes | task_success: 1.000, efficiency: 0.967 |
| 2 | convergence/error_handling | No | task_success: 1.000, efficiency: 0.873 |
| 3 | tool_use/research_calculate | No | task_success: 0.000, efficiency: 0.867 |

## Weighted Scores

| Scenario | Weighted Score |
|----------|----------------|
| classification/sentiment | 0.2815 |
| constraint/json_schema | 0.8158 |
| convergence/error_handling | 0.9638 |
| tool_use/research_calculate | 0.2039 |
