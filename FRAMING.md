# Framing: Rawaj Pay mule detection

## Decision

The decision is which accounts go into the review queue of Dina's five analysts, and which get held before the money leaves Rawaj. Their daily capacity is limited, so the model must rank cases rather than flag everything suspicious. A missed mule means a victim's savings are gone; a wrongly frozen account wastes analyst time and can hurt a real business. The capacity and the cost ratio between these errors are assumptions to confirm with Dina.

## Metric

The main metric is expected cost (`FP * cost_fp + FN * cost_fn`) at the threshold that minimises it, reported with precision@K, where K is the number of cases analysts can review per day. Accuracy is misleading: mules are under 3% of labelled rows in the sample, so a model that flags nothing would still score above 97%.

## Baseline

The baseline is Hares, the current rules engine, measured on the same labelled accounts and the same review capacity. Simpler baselines are flagging nothing, or ranking by one signal such as account age. The comparison uses a time-based split and excludes leaky columns such as `case_closed_at` and `reversal_amount`, which are only known after an investigation.

## Classical ML or LLM

A classical model, such as gradient-boosted trees or logistic regression, is the right fit for structured transfer features with labels: it is fast enough not to slow transfers, runs on Rawaj's own servers, and gives reasons an analyst can use. An LLM is a poor fit for producing the score itself, but a template or small model can turn the top reasons into the Arabic and English explanation compliance needs, if Rawaj's data rules allow it

<!-- Classical model, LLM, a rule, or a mix? Argue from the decision table in the session. -->
