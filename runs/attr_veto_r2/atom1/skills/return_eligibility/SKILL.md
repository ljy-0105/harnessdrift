# 退货资格判断

按顺序检查，任何一步不通过就停下，把不通过的那一条讲给客户：

1. `create_return` 只对 `status == delivered` 生效，其余状态先解释当前进度
2. 查该品类的 `return_window_days`，用 `days_since_delivery` 比较（等于窗口当天仍然可以）
3. 数码品类 `activated == true` 时，无理由退货不成立
4. 窗口为 0 的品类（食品）不支持无理由退货
5. 质量问题、发错货、少件漏发不受窗口限制，且运费由平台承担