# NVDA context glossary

Short definitions of the domain terms used in this implementation's prompts, experiment focus
files, and graduated pattern cues. Use it when writing a study prompt or reviewing a pattern
the agent proposes. It tells you what a term means and why it can move the stock.

> **Only definitions and pre-2025 background belong here. Never add events from 2025 or
> later.** Prompt authors copy text from this file. A 2025 event written here would leak into
> backtest origins that come before it. The training cutoff is explained in
> [`LLM_CUTOFFS.md`](../../../LLM_CUTOFFS.md).

## The company and the stock

| Term | Definition | Why it moves NVDA |
|---|---|---|
| **Data Center segment** | NVDA's reporting segment for accelerators, networking, and systems sold for servers. It overtook Gaming as the main revenue line in 2023. | This is the segment the market prices. Guidance for it is the most important number in an earnings release. |
| **Fiscal calendar** | NVDA's fiscal year ends in late January. For example, FY2025 ran from February 2024 to January 2025. | Headlines about "Q3 FY25" refer to calendar Aug–Oct 2024. Mixing the two up puts events in the wrong window. |
| **Earnings reaction session** | NVDA reports after the US close, in late Feb, May, Aug, and Nov. The price reacts in the next session. | In 2020–2024, 8 of 20 reaction sessions moved ≥7%, against about 4% of sessions overall. This is the biggest predictable source of shocks. See `shock_anchors.py`. |
| **Guidance / beat-and-raise** | Management's revenue forecast for next quarter. "Beat-and-raise" means the quarter beat estimates and guidance went up. | The stock reacts to guidance compared with what investors expected, not to the reported quarter. A beat with weak guidance can still fall. |
| **Gross margin** | Revenue minus cost of goods, as a share of revenue. | It falls during a new product ramp. The market reads a margin drop as either execution risk or a pricing signal. |
| **Split-adjusted price** | Past prices divided by later split ratios: 4:1 in July 2021 and 10:1 in June 2024. | Every series here is adjusted. Raw prices show fake drops of −75% and −90% on split dates. |
| **Shock** (this project) | A move of \|1-session return\| ≥ 7% in either direction. See `paths.SHOCK_THRESHOLD`. | The target event for the shock task and the discovery loop. |

## Demand: the AI capex cycle

| Term | Definition | Why it moves NVDA |
|---|---|---|
| **Hyperscaler** | A very large cloud operator: Microsoft (Azure), Alphabet (Google Cloud), Amazon (AWS), Meta, and Oracle. | Together they buy a large share of NVDA's data-center output. |
| **Capex (capital expenditure)** | Spending on physical assets. Here it mostly means data centers, servers, and accelerators. Hyperscalers report it every quarter and give guidance for the year. | When a hyperscaler raises capex guidance, the market reads it as demand for NVDA, often before NVDA reports. It's usually released in the weeks before NVDA's own earnings. |
| **Semiconductor cycle** | The industry's repeating pattern of shortage, capacity build-out, glut, and inventory correction, historically about 3–4 years. | "Peak cycle" worries, such as capex slowing or customers building inventory, are the standard bear case. They cause sharp de-ratings even when results are strong. |
| **Training vs inference demand** | Training builds a model. Inference runs it for users. | Inference is seen as the more durable, longer-lasting demand. Commentary about the mix changes the narrative on how long demand lasts. |
| **Sovereign AI** | National governments building domestic AI compute. | A demand source separate from the hyperscalers. It is often announced in large, headline-grabbing deals. |

## Supply: product roadmap and manufacturing

| Term | Definition | Why it moves NVDA |
|---|---|---|
| **GPU roadmap** | NVDA's sequence of data-center architectures: Ampere (A100), Hopper (H100/H200), then Blackwell (B100/B200/GB200, announced March 2024). In 2024 NVDA moved to a roughly annual cadence. | Delays in a product transition, or buyers pausing ahead of it, move estimates for the coming quarters. |
| **CUDA** | NVDA's software platform for programming GPUs. | The main reason customers stay with NVDA. Credible alternatives to CUDA are a competition risk. |
| **NVLink / networking** | NVDA's chip-to-chip links plus its InfiniBand and Ethernet networking (Mellanox, acquired 2020). | A fast-growing part of Data Center revenue. It's also why NVDA sells whole racks, not just chips. |
| **TSMC** | Taiwan Semiconductor Manufacturing Co., which manufactures NVDA's leading-edge chips. | NVDA has no factories of its own, so TSMC's capacity, yields, and Taiwan geopolitical risk are all NVDA supply risks. |
| **Yield** | The share of chips on a wafer that work. | Low yields on a new node or package delay supply and hurt margins. |
| **CoWoS** | TSMC's advanced packaging, which puts the GPU die next to HBM memory. It was the main supply bottleneck for AI accelerators in 2023–2024. | News of CoWoS capacity expansion or shortage is a direct read on how much NVDA can ship. |
| **HBM (high-bandwidth memory)** | Stacked DRAM packaged with the GPU. Suppliers are SK hynix, Samsung, and Micron. | HBM supply and qualification problems limit shipments. Memory vendors' results give an early read on accelerator volumes. |

## Policy: export controls

| Term | Definition | Why it moves NVDA |
|---|---|---|
| **Export controls** | US rules restricting the sale of advanced chips and chipmaking tools to some countries, mainly China. They are administered by the Commerce Department's Bureau of Industry and Security (BIS). | China was a meaningful share of Data Center revenue. New rules can remove it quickly, and sometimes they take effect immediately. |
| **October 2022 and October 2023 rules** | The first set of rules restricted A100/H100-class chips to China based on performance thresholds. The update lowered the thresholds and closed the loophole used by the China-specific A800/H800. | These are the template for how a controls headline plays out: an announcement, then guidance cuts, then China-compliant replacement chips. |
| **China-compliant SKU** | A cut-down chip designed to fall under the thresholds, such as the A800, H800, or H20. | Whether such a chip can still be sold decides how much China revenue is left. Every new rule reopens the question. |
| **Entity List** | A BIS list of foreign companies that need a licence to receive US technology. | Adding large Chinese cloud or AI companies to it shuts off customers directly. |

## Competition

| Term | Definition | Why it moves NVDA |
|---|---|---|
| **AMD Instinct** | AMD's data-center accelerators. The MI300X launched in December 2023. | The most direct merchant competitor. Launch events and customer wins are "competitor launch" catalysts. |
| **Custom silicon (ASIC)** | Accelerators that hyperscalers design for themselves: Google TPU, AWS Trainium/Inferentia, Microsoft Maia, Meta MTIA. Broadcom and Marvell often help build them. | If customers move to their own chips, NVDA's addressable market shrinks. Announcements from hyperscalers or Broadcom are read as a risk to NVDA's share. |
| **Intel Gaudi** | Intel's AI accelerator line. | A smaller competitor. Its news rarely moves NVDA on its own. |

## Statistics terms used in cues and gates

| Term | Definition |
|---|---|
| **Base rate** | The share of windows that are shocks, before looking at any news. Precision has to beat this. |
| **Precision** | P(shock \| the pattern matched). |
| **Lift** | Precision divided by base rate. The gate requires ≥ 2.0 on train and ≥ 1.5 on holdout (see `signals.py`). |
| **Trailing volatility** | The std of daily returns over the previous 21 sessions, known at the origin. Shocks cluster when it is high, so a pattern that only tracks volatility is not a news finding. |
