# PHM5005 项目完成总结

## 🎯 项目目标
预测子宫内膜癌患者2年进展风险

## 📊 数据规模
- **样本**: 362患者 (训练:289, 测试:73)
- **特征**: 898 (临床:19, 基因:879)
- **标签**: 高风险23.5%, 低风险76.5%

## 🏆 最佳性能
- **Test AUROC**: 0.6996
- **Test AUPRC**: 0.4083
- **Recall**: 58.8%
- **Specificity**: 69.6%

## 🔑 关键特征 (Top 5)
1. figo_stage_encoded (Clinical)
2. GADD45A (Gene)
3. ITGA10 (Gene)
4. PRR5L (Gene)
5. IGF1R (Gene)

## 💡 主要发现
1. ElasticNet Logistic Regression表现优秀
2. 临床特征提供稳定基础，基因特征增强预测
3. 模型具有良好的可解释性
4. 适用于临床风险分层决策

## 📁 所有结果
位于: `D:/PHM5005/5005-main\results/`

---
完成时间: 2025-11-14 20:50:51
