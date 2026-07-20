# A first test of Prediction 4: mediated one-shot games among LLM agents

Protocol (registered before the full run; pilot n=8 validated schema only):

- One-shot, two-agent, simultaneous COOPERATE/DEFECT. No repetition, no
  reputation, no communication. Subjects instructed to maximize own points.
- Payoffs (x100 of the paper's normalized stage game, beta_kappa = 1.0,
  beta_alpha = 0.6): mutual C: +60 each; unilateral D: defector +D_pts,
  lone cooperator -100; mutual D: -100 each.
- Temptation sweep D_pts in {20, 40, 60, 80, 100}; the patience-free
  boundary (beta_alpha = beta_D) sits at 60. For D_pts < 60, (C,C) and
  (D,D) are both equilibria (mediation = equilibrium selection); for
  D_pts > 60 defection is dominant.
- Conditions: mediator recommendation present/absent. Mediator text:
  independent, non-enforcing registry publishing the same COOPERATE
  recommendation to both agents; cannot punish, reward, or observe.
- Subjects: two current commercial LLMs (Anthropic Claude family: one
  small, one mid-tier), 24 episodes per cell; 5 x 2 x 2 x 24 = 480.
- Outcome: compliance (COOPERATE) rate per cell, Wilson 95% CIs.
- Prediction 4 (paper, Discussion): if agents with beta_alpha > beta_D
  systematically fail to cooperate, C1* is insufficient. Test statistic:
  compliance in patience-free cells (D_pts < 60) under mediation.

Files: episodes.json (raw), analyze_p4.py (analysis), RESULTS_P4.md (output).
