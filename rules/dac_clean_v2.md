`v1` 的主导偏差是干燥样本的系统性低估，最明显出现在 `VpA` 相对 `load` 偏小、且 `v1` 已略高于邻域均值的高拥挤区。另一簇是边界态高估：在近零 `VpA` 的环境 ppm / 干燥条件下，哪怕 `v1` 只在 `nb5m` 附近，真实值仍常更低。中等工况下若 `v1` 已落在邻域统计带内，强修正反而更容易制造新误差。

RULE 1: When `RH=0`、`VpA <= 1.25*load`，且 `v1 >= nb5m + 0.25*nb5s`，上调到 `nb5m + 1.5–5.0*nb5s`。Evidence: ids [1, 11, 12, 18, 19]. Physics: 干燥且胺相拥挤时，反应位点密度会放大化学吸附，基于邻域中心的预测通常偏保守。 Constraints: A✓ B✓ C✓ D✓

RULE 2: When `RH=0`、`ppm` 在 `350–500`、`VpA <= 0.1*load`，且 `v1` 位于 `nb5m - 1.5*nb5s` 到 `nb5m + 0.75*nb5s`，下调到 `nb5m - 1.0–3.0*nb5s`。Evidence: ids [7, 23, 24]. Physics: 极低可达体积会显著压缩有效反应位点数，所以停留在邻域中心附近的预测往往仍偏高。 Constraints: A✓ B✓ C✓ D✓

RULE 3: When `RH=0`、`ppm` 在 `350–500`、`TC` 在 `20–45`、`VpA > 0.1*load`，且 `v1` 位于 `nb5m - 1.5*nb5s` 到 `nb5m + 0.25*nb5s`，只向 `nb5m` 轻微收缩 `0–0.5*nb5s`。Evidence: ids [9, 13, 15, 21]. Physics: 非边界、非高驱动力工况下，模型与邻域统计已基本一致，强修正更容易把低误差样本推坏。 Constraints: A✓ B✓ C✓ D✓