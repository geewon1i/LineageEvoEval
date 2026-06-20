# LineageEvoEval

LineageEvoEval 是一个可分享给组员使用的独立评测框架。不同 baseline 只负责输出已经排好名的因子表达式，本框架统一负责按输入排名截断 topK、test IC 和 Qlib 回测。

重要口径：

- 输入文件中的顺序就是因子排名，从高到低。
- `--select top1` 直接取输入第 1 个因子。
- `--select top5` 直接取输入前 5 个因子。
- `--select top20` 直接取输入前 20 个因子。
- 框架不计算 train / valid IC。
- `--selection-metric valid_ic|train_ic` 只是兼容旧命令的参数，在 test-only 口径下不参与排序、选择或方向处理。
- 默认不根据 test IC 翻转方向，`orientation=1`；如果 baseline 需要反向因子，应直接在输入表达式中写成反向形式。

## 1. 安装

进入项目目录：

```powershell
cd LineageEvoEval
pip install -e .
```

如果当前环境还没有 Qlib：

```powershell
pip install -e ".[qlib]"
```

测试依赖：

```powershell
pip install -e ".[test]"
```

## 2. 配置数据路径

复制配置模板：

```powershell
copy configs\eval.example.toml configs\eval.local.toml
```

在 `configs/eval.local.toml` 填写本机 Qlib 数据路径：

```toml
default_dataset = "csi500"

[datasets.csi500]
provider_uri = "D:/qlib/qlib_data/cn_data"
region = "cn"
market = "csi500"
benchmark = "SH000905"

[datasets.sp500]
provider_uri = "D:/qlib/qlib_data/us_data"
region = "us"
market = "sp500"
benchmark = "SPY"
```

如果本地 S&P500 的 `market` 或 `benchmark` 名称不同，直接改配置文件。

## 3. 输入格式

使用 JSONL，一行一个因子。输入顺序必须是 baseline 已经排好的排名：

```jsonl
{"factor_id": "baseline_a_001", "expression": "Rank($close)", "baseline": "baseline_a"}
{"factor_id": "baseline_a_002", "expression": "TsMean($close, 5)", "baseline": "baseline_a"}
{"factor_id": "baseline_a_003", "expression": "Div(Sub($open, $close), Add(Sub($high, $low), 0.001))", "baseline": "baseline_a"}
```

字段：

- `expression`：必填，因子表达式。
- `factor_id`：可选，不填会自动生成。
- `baseline`：建议填写，用于追踪来源。

表达式采用 LineageEvo 的 AlphaPROBE-style DSL。可用字段：

```text
$open, $high, $low, $close, $vwap, $volume
```

允许的滚动窗口常数：

```text
1, 3, 5, 10, 20, 30, 60
```

允许的算术常数：

```text
0.0001, 0.001, 0.01, 0.0, 1.0, 2.0
```

## 4. 运行

默认使用配置中的 `default_dataset`，模板默认是 `csi500`：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl
```

指定 CSI500：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --dataset csi500 --input inputs\your_factors.jsonl
```

指定 S&P500：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --dataset sp500 --input inputs\your_factors.jsonl
```

取输入排名第 1 个因子：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl --select top1
```

取输入排名前 5 个因子：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl --select top5 --selection-metric valid_ic
```

取输入排名前 20 个因子：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl --select top20 --selection-metric train_ic
```

只算 IC，不跑回测：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl --no-backtest
```

指定输出目录：

```powershell
lineageevo-eval evaluate --config configs\eval.local.toml --input inputs\your_factors.jsonl --output-dir runs\baseline_a_top5
```

## 5. 统一评测设置

默认时间切分：

| split | start | end |
| --- | --- | --- |
| train | 2015-01-01 | 2020-12-31 |
| valid | 2021-01-01 | 2022-04-30 |
| test | 2022-05-01 | 2026-04-30 |

默认 label：

```text
Ref($close, -2) / Ref($close, -1) - 1
```

默认 IC 方法：

```text
spearman
```

组合因子口径：

1. 直接使用 baseline 输入表达式的方向，`orientation=1`。
2. 在 test 区间每日横截面 z-score。
3. 对选中因子等权平均，得到组合 signal。
4. 用组合 signal 计算 test IC / ICIR，并用于 Qlib 回测。

默认回测参数：

```toml
[backtest]
enabled = true
account = 100000000
topk = 50
n_drop = 5
risk_degree = 0.95
```

## 6. 输出文件

每次运行生成一个独立输出目录，默认在 `runs/` 下。

| 文件 | 内容 |
| --- | --- |
| `config_snapshot.json` | 本次实际生效配置 |
| `factor_evaluations.csv` | 所有输入因子的表达式校验状态和 Qlib 表达式 |
| `selected_factors.csv` | 按输入顺序截断得到的 topK 和方向 |
| `test_ic_results.csv` | 被选中单因子的 oriented test IC / ICIR |
| `composite_test_ic_results.csv` | 组合因子的 test IC / ICIR |
| `backtest_summary.csv` | Qlib 回测摘要 |
| `backtest_daily_report.csv` | Qlib 回测逐日结果 |
| `run_summary.json` | 运行摘要和失败原因统计 |

## 7. 常见问题

### provider_uri is empty

说明还没有在 `configs/eval.local.toml` 填写对应数据集路径。

### provider_uri does not exist

说明路径不存在。检查盘符、目录名和 Qlib 数据位置。

### pyqlib is not available

当前 Python 环境不能 import `qlib`。请切换到已有 Qlib 的环境，或安装：

```powershell
pip install -e ".[qlib]"
```

### factor_evaluations.csv 里有 failed

说明对应表达式校验或 Qlib 计算失败，原因会写在 `failure_reason` 列。

如果 failed 出现在输入前 K 个因子中，框架不会用后面的因子补位；这保证 `top1/top5/top20` 始终表示输入排名截断。

### 为什么不自动按 test IC 翻方向？

用 test IC 的符号翻方向会引入测试集信息泄漏，所以框架默认保持 baseline 提供的表达式方向。需要反向时，请在表达式里直接写反向形式，例如用 `Sub(0.0, 原表达式)`。

## 8. Python 接口

```python
from lineageevo_eval import EvaluationOptions, evaluate_file

result = evaluate_file(
    "inputs/your_factors.jsonl",
    EvaluationOptions(
        config_path="configs/eval.local.toml",
        dataset="csi500",
        select="top5",
        selection_metric="valid_ic",
        run_backtest=True,
    ),
)

print(result.output_dir)
print(result.summary)
```
