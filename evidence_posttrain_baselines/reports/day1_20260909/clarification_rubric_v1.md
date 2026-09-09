# 澄清有效性评分规则 v1（已冻结：模型辅助评分）

状态：已校准 3 个 dev 家族的 9 条轨迹，冻结于首次 test 内容审阅之前。适用于固定标准回复条件下的离线评测。

## 评分身份与材料

本次助手可完成模型辅助的逐条语义评分，不能标记为人工评分。若用户或另一位人工评审随后复核，分别记录 reviewer、覆盖数、分歧与裁决。独立人工验证当前为未完成。

评分材料隐去模型、run、seed、历史指标，保留随机 blind_id；映射单独保存。展示 query、初始可见证据、模型真实首轮、标准用户回复、gold/恢复槽元数据及最终输出。必须区分“模型初轮可见内容”和“裁判额外参考”。隐藏身份不消除同一评分助手对研究假设的先验了解。

## 标签

每项使用 true / false / null；null 代表未判定或不适用，另以 judgment_status 标明 pending 或 complete。不得用 Python bool('false') 等字符串强制转换进行汇总。

| 字段 | true 的条件 |
| --- | --- |
| action_correct | 使用统一解析口径后，MISSING 为 REQUEST_INFORMATION，CONFLICT 为 REQUEST_CONFIRMATION |
| gap_or_conflict_correct | 请求实际缺失的必要关系，或准确识别同一必要槽的两项互斥断言 |
| evidence_faithful | 对已有证据的实质陈述有可定位支持；没有把 query 条件或裁判隐藏恢复事实说成已获证实 |
| specific_single_request | 请求目标具体且集中于一个可恢复槽，不是泛化“更多信息”或多个无关问题 |
| unsupported_answer_preset | 输出无依据地断言或预设最终答案/缺失关键关系；这是负向标签 |
| user_reply_fit | 标准回复直接回答模型请求，并提供足以恢复任务的必要信息 |
| final_answer_correct | 沿用核验后的严格动作+答案别名匹配口径，续轮 ANSWER 且匹配正确 |

## 判定细则

1. 不要求固定模板、固定词汇或必须先总结证据。未总结证据不扣分；有总结则检查忠实性。
2. MISSING 允许答案实体仍在剩余证据中。合理引用实体不自动判泄露；无依据断言该实体满足缺失关系才计入答案预设。
3. 对未证实候选的试探性确认，需要结合协议判断是否夹带答案猜测。有争议则 pending，不为某个 run 特设例外。
4. CONFLICT 可合理列出两项争议值，请用户裁决，不能因出现 gold 字符串扣泄露分。
5. 若只列两个人名，却把真正争议的关系说错，则 gap_or_conflict_correct=false。
6. 请求更多信息但没有具体槽，即使标准回复恰好提供答案，也不算有效澄清。
7. 请求了错误属性，而标准回复补了真正缺口，user_reply_fit=false；最终答对仍不足以严格成功。
8. 用户回复对齐不表示实际用户一定能回答；本指标只约束现有固定回复协议。
9. 初轮硬答、解析失败、无请求或续轮缺失均使严格 e2e+=0，分母保留。
10. 每个 false/pending 标签写明原因，并引用最短必要原文；通过记录也保存关键槽依据以便复核。

## 聚合

clarification_valid = gap_or_conflict_correct AND evidence_faithful AND specific_single_request AND NOT unsupported_answer_preset。

strict_e2e_plus = action_correct AND clarification_valid AND user_reply_fit AND final_answer_correct。

只要任一必要项明确失败，严格分为 0；全部必要项明确通过才为 1；其余记录保留 pending，并计算上下界。不以 pending 删除分母。

record 级：对全部扰动记录平均。source 级：先平均同源各家族分数，再平均 source。FULL 按 source 去重，与扰动记录级指标分开报告。不同 seed 保留独立运行汇总，不能当作更多独立 source。


## dev 校准后的解释细则

- 评价完整澄清，而非只截取最后一句。例如前句已明确两端城市，后句的 cities served 可按端点请求理解。
- 允许一个事实槽由两个值组成（如路线两端），不因此视为多问。
- 时间/关系的简写结合 query 与完整澄清解释；若确实请求了另一时间或关系，仍失败。
- 没有请求的输出，澄清内容字段记 null/N/A，严格分因动作失败为 0，不混入待裁决。
- 可能的源数据实体消歧问题另列数据质量备注，本轮保持冻结 gold 口径，不临时筛除不利样本。

## 冻结记录

时间：2026-09-09T04:51:33.917050+00:00

样例 DEV-001 至 DEV-009，涉及 PEI 1873/1874、Wetzel 的 Bamberg 县关系、Sunset Limited 端点。九条在冻结协议内均通过；Bamberg 同名实体跨文档链接有消歧疑点，记为数据审计候选。另按草案复核泛化请求、问错属性、预设缺失答案三类规则反例，不计作真实 dev 样本。

评分类型：当前助手的模型辅助、隐藏 run 身份评分；人工复核=0。并非完全独立人工盲法。测试内容尚未审阅，身份映射独立保存。
