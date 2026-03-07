# A 股次日候选股筛选器

这是一个面向 A 股的轻量级选股脚本：

- 默认从 `stock_picker/data/universe.default.csv` 读取股票池
- 默认股票池是 12 只高流动性核心股票，优先保证定时任务稳定性
- 用 AkShare 拉取历史日线
- 根据技术面和成交额特征训练一个随机森林回归模型
- 输出次日预期收益率最高的前 2 只股票
- 如果实时行情接口可用，会在 14:30 额外叠加盘中量比/换手率信息做微调

> 重要提示：这不是投资建议，也不能保证“第二天一定上涨”。它本质上是一个“概率排序器”，只适合做研究和辅助决策。

## 适合你的几个开源项目

如果你想要比这个脚本更完整的框架，比较值得看的有：

1. **Microsoft Qlib**  
   偏 AI 量化研究平台，因子工程、模型训练、回测都比较完整，适合做机器学习选股。

2. **FinRL**  
   偏强化学习交易研究，适合探索 RL 在资产配置和择时里的应用。

3. **vn.py / VeighNa**  
   国内生态比较成熟，实盘和事件驱动框架更强，也有机器学习扩展模块，适合后续从研究走向交易系统。

4. **Backtrader**  
   更偏回测框架，本身不是 AI 平台，但策略验证和策略组合非常方便，适合和自定义模型配合。

如果你问我“哪个最适合从零开始”：

- 想做 **AI 选股研究**：优先 `Qlib`
- 想做 **国内可扩展的交易框架**：优先 `vn.py`
- 想做 **轻量原型和回测**：这个仓库里的脚本 + `Backtrader` 就够快

## 安装

```bash
python3 -m pip install -r stock_picker/requirements.txt
```

## 运行

默认输出 2 只候选股：

```bash
python3 stock_picker/a_share_picker.py --top-n 2
```

如果你只想用日线，不依赖实时接口：

```bash
python3 stock_picker/a_share_picker.py --top-n 2 --disable-live
```

如果你修改了股票池，想强制刷新缓存：

```bash
python3 stock_picker/a_share_picker.py --top-n 2 --refresh-cache
```

如果你把股票池扩成了几十上百只，建议先限制请求规模：

```bash
python3 stock_picker/a_share_picker.py --top-n 2 --max-symbols 20 --request-pause-seconds 1.2 --disable-live
```

## 输出文件

脚本运行后会写出：

- `stock_picker/output/latest_picks.json`
- `stock_picker/output/latest_picks.csv`
- `stock_picker/output/daily_picks_YYYYMMDD.json`
- `stock_picker/output/daily_picks_YYYYMMDD.csv`

JSON 中会包含：

- 生成时间
- 是否使用了实时行情增强
- 基础验证指标
- 当日候选股及预期次日涨跌幅

## 默认股票池

默认股票池不是“全市场”，而是一个较稳的高流动性核心池，放在：

```text
stock_picker/data/universe.default.csv
```

你可以把它替换成自己的自选股、行业池、沪深 300、中证 500 等。  
只要是 CSV 并包含 `symbol` 列即可，例如：

```csv
symbol,name
600519,贵州茅台
300750,宁德时代
000858,五粮液
```

## 定时任务：每天下午 14:30 运行

Linux `crontab` 示例：

```bash
crontab -e
```

加入下面这行：

```cron
30 14 * * 1-5 /usr/bin/python3 /workspace/stock_picker/a_share_picker.py --top-n 2 --disable-live >> /workspace/stock_picker/output/cron.log 2>&1
```

如果你的环境可以稳定访问实时行情，就去掉 `--disable-live`。
如果你改成更大的股票池，建议同时显式指定 `--max-symbols` 和 `--request-pause-seconds`。

## 模型思路

脚本当前使用的是一个相对保守的方案：

- 特征：
  - 1/3/5/10/20 日收益率
  - 收盘价相对均线偏离
  - 波动率
  - 成交额放大倍数
  - 换手率放大倍数
  - 价格在 20/60 日区间中的位置
  - 开盘缺口、实体、上下影线
- 标签：
  - `次日收益率 = 下一交易日收盘 / 当日收盘 - 1`
- 模型：
  - `RandomForestRegressor`

如果实时行情抓取成功，还会额外把以下盘中信息作为小幅打分修正：

- 量比
- 换手率
- 当前涨跌幅绝对值

## 局限

这个脚本有几个必须明确的局限：

1. 默认股票池不是全 A。
2. 预测结果只是一种统计意义上的排序，不是确定性结论。
3. 没有接入财报、公告、资金流、板块联动、分钟级历史数据。
4. “14:30 盘中预测”目前是 **日线模型 + 盘中弱增强**，不是完整的分钟级机器学习模型。
5. 第三方免费行情接口会有节流或断连，所以脚本做了缓存、限量和自动降级。

如果你后面想继续升级，推荐路线是：

1. 把股票池换成你自己的全市场候选池
2. 增加行业、财务、资金流和情绪特征
3. 增加 walk-forward 回测
4. 增加通知模块（企业微信/钉钉/Telegram）
5. 把模型升级成 LightGBM / XGBoost / Qlib 因子流程
