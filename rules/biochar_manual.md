## Biochar Physics-Informed Correction Rules

R1' Threshold Response Rule:
When the activation agent is alkaline (KOH/NaOH/K2CO3), Act_Mass >= 3.0, Act_Temp >= 700,
and the closest five neighbors have median CO2 uptake > 4.0, lift toward
nb5_mean + 0.5 * nb5_std.

R2' Optimal KOH Window Rule:
When Act_Agent is KOH, Act_Temp is in [700, 850], Vnarrow > 0.4, and the XGBoost reference
is above the selected neighborhood mean, lift by 0.2 * nb_std.

R3' Kinetic Penalty Rule:
When Act_Ramp > 10 and the tentative prediction exceeds nb_mean + 0.5 * nb_std, pull it back
to nb_mean + 0.2 * nb_std.

R4' Adsorption Condition Adjustment:
If Ads_Temp < 0, lift by 15%. If Ads_Temp > 35, reduce by 15%.
If CO2_PP < 0.2, reduce by 30%. If CO2_PP > 1.0, lift by 20%.

Apply R3 first, then R1/R2, and finally R4. Cap any one rule adjustment at 0.6 mmol/g in
absolute value and keep the final adjustment within 1.0 mmol/g of the selected local anchor.
