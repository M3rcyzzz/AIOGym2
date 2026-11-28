import { Section } from "@/components/site"
import { useLang } from "@/lib/i18n"
import { NetworkDiagram, TrainingDiagram } from "@/sections/SeqDiagram"

/* Network design and training method (user 11-10; expanded with visual diagrams and plain engineering prose 11-27).
   Written from the code in service on main:
   - Model: aio/seqnet.py (SeqAIO v2: width 768, 12 layers, 12 heads, QK-Norm, RoPE, dense multi-lag cross-channel
     attention across 1/3/8/16/24/36/48/64 steps, 64-dim multi-timescale token EMA memory fed back to Layer 0,
     16-dim bilinear pairwise edge memory injected into all 12 layers' channel attention, width-scaled direct action head).
   - Input packet: aio/interface.py.
   - Trainer: aio/train.py --seq (pure real control cost through the differentiable simulator, zero imitation,
     zero auxiliary losses, 64-step truncated BPTT, training-only 4-head 64–256-step dead-time Smith-predictor
     privileged critic, reversal-only anti-chatter smoothing, per-plant gradient balancing, Muon + AdamW + EMA). */

type Item = [string, string]
type Block = { h: Item; items: Item[] }

const BLOCKS: Block[] = [
  { h: ["1 · Training ground: thousands of diverse random plants", "1 · 训练场：上千台风格各异的随机装置"], items: [
    ["Each training run simulates hundreds to thousands of generated industrial plants in parallel, every single one with distinct dynamics.", "每次训练同时并行推进数百至数千台随机生成的工业过程装置，每台装置的动态特性都各不相同。"],
    ["Plant sizes: the base training size has 1–5 manipulated variables (valves, MVs), up to 12 controlled variables (CVs, including up to 4 zone variables held inside limits), 2 measured disturbances (DVs), and 2 unmeasured loads. Larger chained multi-unit plants scale up to 10 valves (24×10) and 20 valves (48×20).", "装置规模：基础档包含 1–5 个操纵变量（阀门，MV）、最多 12 个被控变量（CV，其中最多 4 个是只需不越上下限的区间量）、2 个可测扰动（DV）和 2 个不可测负荷；更大规模的串联多单元装置扩展至 10 阀（24×10）与 20 阀（48×20）。"],
    ["Dynamics and dead time: time constants range from 2.5 to 90 control steps; pure dead time spans from near zero up to 4 times the time constant; 60% of plants place valves and sensors spatially so cross-coupling arrives 2–13 times later than a loop's own response; 12% of CVs are 2.5–10 times faster than the rest of the unit; a quarter carry a slow heat, holdup, or recycle store of 40–200 steps.", "响应快慢与纯滞后：时间常数覆盖 2.5 到 90 个控制步；纯滞后（死时间）从接近零到时间常数的 4 倍；六成装置考虑阀门与传感器的空间距离，跨回路耦合比本回路晚到 2–13 倍；12% 的被控量比主体快 2.5–10 倍；四分之一装置带有 40–200 步的慢蓄存（热容、大持液量或物料循环）。"],
    ["Multivariable coupling: spans decoupled, cascade, strongly interacting, separation, blending, and sparse structures; ill-conditioned plants reach relative gain array (RGA) values up to 20; 10% deliberately publish a non-optimal loop pairing.", "多变量耦合：覆盖解耦、串级、强交互、分离、调和与稀疏六类典型结构；病态耦合装置的相对增益阵（RGA）最高达 20；一成装置刻意给出非最优的常规回路配对。"],
    ["Response shapes and nonlinearity: includes first-order, overdamped, inverse-response (non-minimum phase), lead, and underdamped dynamics; integrating variables (such as tank levels) and slowly unstable variables; curved steady-state gains; in 35% of nonlinear plants, all time constants and dead times vary dynamically with throughput.", "响应形态与非线性：涵盖一阶、过阻尼、反向响应（非最小相位）、超前与欠阻尼；包含积分对象（如液位）与缓慢不稳定对象；包含弯曲增益曲线；35% 的非线性装置中，全部时间常数与纯滞后都会随处理量动态缩放。"],
    ["Field imperfections: operator limits, valve rate limits, positioner lag, stiction, nonlinear flow characteristics, drifting disturbances, sensor drift, measurement noise, quantization, slow sample-and-hold analyzers, spikes, and failed readings. Among multi-valve plants, 28% include spare valves that should rest at a nominal position.", "现场非理想因素：涵盖操作员限位、阀门速率限制、定位器滞后、阀粘滑、流量特性曲线、持续漂移扰动、仪表零漂、测量噪声、量化误差、采样保持型分析仪、尖峰坏值等；多阀装置中 28% 带有平时需停在静置位的富余阀门。"],
  ]},
  { h: ["2 · Long operating stays and realistic process events", "2 · 长驻工况与过程扰动事件"], items: [
    ["Half of the plants run across 8 consecutive segments of 720 control steps with both plant state and network memory preserved; the other half start fresh each segment.", "一半装置连续运行 8 段、每段 720 个控制步，全程保留装置物理状态与模型记忆；另一半每段切换到一台全新装置。"],
    ["Each segment contains roughly 8 process events (65% gradual ramps over 5–120 steps): setpoint changes, shifts in plant gain or dynamics, unmeasured load changes, measured disturbance changes, constraint limit shifts, and priority changes.", "每段包含约 8 个过程事件（其中 65% 是持续 5–120 步的渐变过程）：设定值切换、装置自身增益或滞后漂移、不可测负荷变化、可测扰动变化、上下限调整与优先级切换。"],
    ["About 15% of events push the plant into a constrained operating regime where not every setpoint can be held simultaneously (unreachable setpoint, excessive load, tightened valve or zone limits, a valve taken offline, or a spare valve hitting saturation); the controller must yield lower-priority targets to protect higher-priority ones.", "约 15% 的事件会将装置推入目标无法全部兼顾的受限工况（如设定值超出可达范围、负荷过大、阀门或区间限值收紧、某阀门切出手动、富余阀顶满），控制器必须按优先级主动让步。"],
    ["Over multi-segment stays, equipment gradually wears (without maintenance, the weakest valve's effective gain drops to a median of 0.60 by segment 8); 35% of later plant shifts represent maintenance recoveries.", "长周期运行中设备会逐渐老化（不检修时，第 8 段最弱阀门的有效增益中位数降至 0.60）；后续装置特性变化中 35% 模拟检修恢复。"],
  ]},
  { h: ["3 · The deployed model: one pure end-to-end neural network", "3 · 部署模型：单一纯端到端神经网络"], items: [
    ["About 128 million active runtime parameters (768 hidden width, 12 stacked layers, 12 attention heads). Every controlled variable, measured disturbance, and valve is encoded as one 768-dim token using shared weights, so one model works natively across any number of loops.", "约 1.28 亿活跃运行参数（特征宽 768、12 层主干、12 个注意力头）。每个被控量、可测扰动和阀门都由共享编码器映射为 768 维向量，因此同一套权重无需改结构即可适配任意回路数量。"],
    ["Each layer combines four blocks: causal temporal attention (over the past 128 steps with rotary positional embeddings and QK normalization), multi-lag cross-channel attention (directly inspecting all channels 1, 3, 8, 16, 24, 36, 48, and 64 steps earlier to cover long dead times), simultaneous cross-channel attention, and a feed-forward network.", "每层串联四个模块：因果时间注意力（带旋转位置编码与 QK 归一化，回看自身 128 步历史）、多滞后跨通道注意力（直接跨越 1、3、8、16、24、36、48、64 步查看全部通道以覆盖长纯滞后）、同刻跨通道交互注意力，以及前馈网络。"],
    ["Two persistent memory pathways extend context far past 128 steps: a 64-dim per-channel multi-timescale exponential moving average (forgetting over 4 to 2,048 steps) that feeds directly back into Layer 0, and a gain-aware multi-scale causal state-space + Gated DeltaNet pairwise associative memory that tracks online (CV, MV) causality and physical gain sensitivity, injects 8-scale in-flight pipeline displacement into Layer 0 (a Neural Smith Predictor), biases all 12 layers' channel & lag attention, and passes 1-hop decoupling messages.", "两条跨步长记忆支路将感知范围延展到 128 步之外：一是每个通道自带 64 维多尺度指数移动平均记忆（遗忘周期 4–2,048 步）并回馈第 0 层；二是增益感知的 8 阶因果状态空间与 Gated DeltaNet 成对联想记忆，在线累积多滞后因果与物理增益量级，把 8 尺度在途管道位移注入第 0 层构成神经网络版 Smith 预测器，同时偏置全部 12 层通道与滞后注意力并直连 1 跳解耦消息。"],
    ["Each valve's next move is read directly from its own output token through width-aware scaling, a multiplicative log-gain precision gate (spanning three orders of magnitude from high-gain micro-moves to sluggish loops), and rate-limit bounding. There is no underlying PID controller, no recursive least-squares identifier, and no online optimization solver.", "每个阀门的下一步动作增量直接从其自身输出向量经宽度缩放、横跨三个数量级的微幅精调门控（兼顾高增益静默微调与低增益大步幅）并限幅读出。网络内部没有任何公式，底层不套 PID，也没有最小二乘辨识或在线优化求解器。"],
  ]},
  { h: ["4 · Training: real control cost through a differentiable simulator", "4 · 训练方法：真实代价穿过可微仿真器端到端反传"], items: [
    ["Zero imitation learning: real closed-loop control cost over a 64-step window (tracking error outside band, zone limit violations, and trips) is backpropagated directly through the differentiable physics simulator to every valve move, paired with light sensitivity-scaled exploration noise so ultra-high-gain dead-time loops are never drowned out. A tiny training-only 75-parameter linear probe (removed at deployment) supervises the DeltaNet memory strictly on unpublished and hidden couplings for presence, sign, and physical gain scale.", "零模仿学习：把 64 步滚动窗口里的真实闭环控制代价（超带宽偏差、区间越限与失稳跳停）直接穿过可微物理仿真器反向传播到每一步阀门动作，并配合按通道灵敏度自动缩放的轻量化探索噪声（避免淹没高增益纯滞后回路的微小步幅）；另设一个仅训练期使用的 75 参数线性探针（部署时移除），严格在未告知跨通道与盲控配对上监督存在性、正反作用与物理增益量级。"],
    ["Long-horizon effects past the 64-step window are guided by a training-only privileged value network (~2 million parameters, completely removed at deployment). It sees exact noiseless simulator states—including in-flight moves inside delay pipelines and Smith-predictor-style steady-state error estimates—and predicts realized future discounted cost across four fixed horizons (64, 128, 192, and 256 steps).", "64 步窗口之后的远期代价由仅训练期使用的特权价值网络（约 200 万参数，部署时完全移除）引导。它读取仿真器内部无噪声真值——含延迟管道中尚未抵达的在途动作与 Smith 预测器式终态误差——并用未来真实回放代价监督预测 64、128、192、256 步四个定长远期代价。"],
    ["Three quarters of training episodes publish the loop pairing and action direction; one quarter hide both so the model can probe valves before the first event. One fifth of episodes include a stuck valve or failed sensor.", "四分之三的训练回合公布常规回路配对与作用方向；四分之一完全隐藏，允许模型在首个事件前自主试阀摸底。五分之一回合包含阀门卡死或传感器失效。"],
    ["Three plant sizes (12×5, 24×10, 48×20) train jointly in large parallel batches; reversal-only smoothing suppresses valve chatter without slowing monotonic moves; per-plant cost normalization with a 0.5 floor keeps hard long-delay plants from losing gradient share; hybrid Muon + AdamW optimization with cosine decay and EMA weight averaging stabilizes updates.", "三种规模装置（12×5、24×10、48×20）同批大批量联合训练；仅惩罚来回换向的反向平滑正则在抑制阀门抖动的同时不拖慢单调响应；带 0.5 下限的单样本代价归一化确保高难度长滞后装置获得足够梯度权重；采用 Muon + AdamW 混合优化器、余弦学习率退火与指数移动平均权重（EMA）稳定收敛。"],
  ]},
  { h: ["5 · Evaluation benchmarks and scoring scale", "5 · 考试体系与统一打分标准"], items: [
    ["Raw cost counts every step after the first event that a tracked CV spends outside its allowed band plus every step a zone CV violates a limit, evaluated on true noiseless plant values; any plant trip counts the entire remaining window as out-of-band.", "原始控制成本统计第一个事件之后，所有跟踪被控量偏离允许带宽的累计步数加上所有区间量越限的累计步数，全部按装置真值计算；一旦发生失稳跳停，剩余窗口全部计为越限。"],
    ["Unified 0–100 score scale: 0 points = leaving all valves frozen; 100 points = per-episode envelope of the best reference controller, anchored by oracle MPC (which always knows the exact true plant equations and internal state, lacking only future events and noise).", "统一 0–100 分制：0 分代表所有阀门保持不动；100 分代表每个回合最优参照控制器的包络线，以 oracle MPC 为锚点（始终掌握装置精确模型与内部真值，只不知晓未来事件与噪声）。"],
    ["Reference baselines on the same test sets: self-tuning PID (one PID loop per published pairing, tuned from the exact starting step response), step test + static-model MPC (runs a ±5% identification step test with PI in the loop while scored, then deploys the identified MPC), and static-model MPC (handed the exact initial model for free at the start without updating after process changes—the practical upper reference).", "同批测试集上的三条工程参照线：自整定PID（按起始精确阶跃响应一次性整定的单回路 PID）、阶跃测试 + 静态模型MPC（带扰动信号边辨识边计分再切入 MPC），以及静态模型MPC（开局免费获得精确初始模型、仅靠仪表校正且后续不更新——作为强参照上界）。"],
    ["Test suites: a fixed 256-plant unseen validation set (12×5), larger fixed unseen sets (24×10 and 48×20), long 8-segment stay tests, and zero-shot mechanistic process plants never seen during training.", "多维度考题：固定 256 台未见随机装置验证批（12×5）、更大规模固定未见批（24×10 与 48×20）、8 段连续长驻测试，以及训练全程从未接触过的经典工业机理装置零样本考试。"],
  ]},
]

const MECH: [Item, Item, string][] = [
  [["Exam benchmarks (15 classic processes)", "经典工业机理装置考题（15 类）"], ["Quadruple tank, pH neutralizer, Van de Vusse reactor, exothermic CSTR, bioreactor, evaporator, furnace, gas network, heat-exchanger network, distillation column; building heating, centrifugal compressor, reactor–separator–recycle, ammonia converter, fired heater", "四容水箱、pH 中和槽、Van de Vusse 反应器、放热连续搅拌釜（CSTR）、生物反应器、蒸发器、工业加热炉、燃气管网、换热网络、精馏塔、建筑供暖、离心压缩机、反应–分离–循环系统、合成氨转化炉、管式加热炉"], "15"],
  [["Large-scale industrial processes (6 benchmarks)", "大型复杂机理装置考题（6 类）"], ["Tennessee Eastman Process (TEP, two scales), BSM1 activated-sludge wastewater plant, and three rigorous distillation columns (benzene–toluene, methanol–water, debutanizer)", "田纳西–伊斯曼过程（TEP，两种规模）、BSM1 活性污泥污水处理厂，以及三座严格机理精馏塔（苯–甲苯、甲醇–水、脱丁烷塔）"], "6"],
  [["Strictly held out (exam only, never trained)", "永久留出装置（仅作考题，永不进训练）"], ["Two industrial boilers (one validated on field data), Williams–Otto reactor, MMA polymerization, crystallizer, vinyl acetate plant, GasLib-24 gas network, IFAC 2018 refrigeration, Badgwell fired heater, plus external BOPTEST building heating", "两类工业锅炉（其一对标现场数据）、Williams–Otto 反应器、MMA 聚合反应、结晶器、醋酸乙烯装置、GasLib-24 燃气管网、IFAC 2018 制冷系统、Badgwell 加热炉，另含外部 BOPTEST 建筑供暖"], "9"],
]

const TERMS: Item = [
  "Common terms: a controlled variable (CV) is a process reading to regulate (temperature, level, composition, pressure); a manipulated variable (MV) is a valve or actuator the controller moves; the allowed band is the target tolerance around a setpoint; a zone variable only needs to stay within upper/lower limits; a measured disturbance (DV) is a measurable upstream change no valve controls; pure dead time is the transport delay before a valve move shows up on a sensor. References: self-tuning PID (standard industrial loop tuning), static-model MPC (constrained model predictive control with a fixed initial model), oracle MPC (ideal upper bound knowing the true plant state; unavailable in real plants).",
  "常见术语说明：被控变量（CV）是需要稳住的工艺读数（如温度、液位、成分、压力）；操纵变量（MV）是控制器直接调节的阀门或执行器；允许带宽是设定值上下可接受的偏差范围；区间量只需保持在上下限内而不追固定点；可测扰动（DV）是能测到但不受阀门影响的外部负荷变化；纯滞后（死时间）是指阀门动作后传感器尚未看到响应的传输延迟。参照控制器：自整定PID（按标准工业阶跃响应整定的单回路 PID）、静态模型MPC（持有固定初始模型的带约束模型预测控制）、oracle MPC（掌握装置实时真相的理想理论上限，真实工厂不可得）。",
]

const IO_IN: { h: Item; items: Item[] }[] = [
  { h: ["Each controlled variable (CV), 12 numbers", "每个被控变量（CV），12 个无量纲数"], items: [
    ["Current deviation from setpoint, divided by its allowed band", "当前读数与目标值的偏差（以允许带宽为单位）"],
    ["Reading change since the previous control step", "相比上一步的读数变化量"],
    ["Lower and upper constraint limits relative to the target", "相对目标的上下约束限位（区间量只给限位、无目标）"],
    ["Mode (setpoint tracking vs. zone constraint) and priority weight", "控制类型（目标跟踪或区间约束）与重要性权重"],
    ["Recent setpoint shift, analyzer sample age, and step interval", "设定值刚发生的变动、分析仪样本延迟与控制周期"],
  ]},
  { h: ["Measured disturbances (DV)", "可测扰动（DV）"], items: [
    ["Feed flow, upstream load, or ambient readings encoded in the same 12-number format and flagged as uncontrollable by valves", "进料流量、上游负荷或环境温度等前馈读数，采用同样的 12 维格式，并标明不受任何阀门控制"],
  ]},
  { h: ["Each manipulated variable / valve (MV), 6 numbers", "每个操纵变量 / 阀门（MV），6 个无量纲数"], items: [
    ["Current valve position within operator limits (−1 at low limit, +1 at high limit)", "当前阀位在操作员限位内的归一化开度（下限 −1，上限 +1）"],
    ["Previous valve move and maximum allowed move per step (rate limit)", "上一步实际动作量，以及单步允许的最大速率限制"],
    ["Action direction: whether opening raises or lowers its paired CV (hidden in 25% of episodes)", "正反作用方向：开大阀门会使配对读数上升还是下降（25% 回合隐藏）"],
  ]},
  { h: ["Loop pairing table", "回路配对表"], items: [
    ["Which valve is conventionally paired with which CV (optional commissioning hint; hidden in 25% of episodes)", "常规工程上哪个阀主管哪个被控量（投运参考提示，25% 回合完全隐藏）"],
  ]},
]

const IO_NOT: Item[] = [
  ["The plant mathematical model, transfer functions, or physics equations", "装置数学模型、传递函数或物理方程"],
  ["Commissioning step tests or impulse/step response curves", "投运阶跃测试数据或阶跃响应曲线"],
  ["Plant type, equipment identity, variable names, or physical units", "装置类型、设备编号、变量物理名称或量纲单位"],
  ["Any future setpoints, disturbances, or faults", "未来将发生的设定值切换、扰动或设备故障"],
]

function IODiagram({ L, tr }: { L: (t: Item) => string; tr: (en: string, zh: string) => string }) {
  return (
    <div className="rounded-[10px] border bg-card p-4 sm:p-5">
      <h4 className="mb-3 text-base font-semibold">{tr("Interface: what the model receives and outputs each step", "接口边界：模型每一步拿到什么、给出什么")}</h4>
      <div className="grid items-stretch gap-3 lg:grid-cols-[1.6fr_auto_1fr]">
        <div className="grid gap-2 sm:grid-cols-2">
          {IO_IN.map((g, i) => (
            <div key={i} className="rounded-[8px] border bg-surface p-3">
              <p className="mb-1 text-sm font-semibold">{L(g.h)}</p>
              <ul className="list-disc space-y-1 pl-4">{g.items.map((t, j) => <li key={j} className="text-[13px] leading-relaxed text-foreground/80">{L(t)}</li>)}</ul>
            </div>
          ))}
        </div>
        <div className="flex flex-col items-center justify-center gap-2 py-1">
          <span aria-hidden className="text-xl text-muted-foreground lg:hidden">↓</span>
          <div className="rounded-[8px] border border-[var(--fx-black)] bg-[var(--fx-black)] px-4 py-3 text-center text-white">
            <p className="text-sm font-semibold">{tr("End-to-end model", "端到端控制模型")}</p>
            <p className="num text-[13px] text-[var(--fx-lime)]">{tr("~128M active params", "约 1.28 亿活跃参数")}</p>
            <p className="text-[13px] text-white/75">{tr("Reads only instrument history", "仅凭历史仪表数据决策")}</p>
          </div>
          <span aria-hidden className="text-xl text-muted-foreground lg:hidden">↓</span>
        </div>
        <div className="flex flex-col gap-2">
          <div className="rounded-[8px] border-2 border-[var(--fx-black)] p-3">
            <p className="mb-1 text-sm font-semibold">{tr("Output: direct valve increments", "输出：每个阀门的下一步动作增量")}</p>
            <p className="text-[13px] leading-relaxed text-foreground/80">{tr("One move increment per valve, bounded by its rate limit. No PID loop underneath—it drives every actuator directly.", "每个阀门输出一个控制增量，严格限幅在单步速率限制以内。底层没有 PID，直接驱动阀门。")}</p>
          </div>
          <div className="rounded-[8px] border border-dashed p-3">
            <p className="mb-1 text-sm font-semibold">{tr("Explicitly never given", "明确不给的信息")}</p>
            <ul className="list-disc space-y-1 pl-4">{IO_NOT.map((t, j) => <li key={j} className="text-[13px] leading-relaxed text-foreground/80">{L(t)}</li>)}</ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export function Method() {
  const { tr } = useLang()
  const L = (t: Item) => tr(t[0], t[1])
  return (
    <Section id="method" index="02" title={tr("Model and training architecture", "模型结构与训练方法")}>
      <p className="mb-4 max-w-[900px] text-sm leading-relaxed text-foreground/85">
        {tr(
          "One pretrained end-to-end neural network reads the dimensionless instrument history of any industrial plant and writes valve moves directly, without any control formula, online system identification, or optimization solver inside. It trains purely by backpropagating real closed-loop control cost through a differentiable simulator across thousands of diverse plants, and is evaluated zero-shot on unseen plants.",
          "我们用一张预训练的端到端神经网络直接充当通用先进过程控制器（APC）：每一步读取任意工业装置的无量纲历史仪表数据，直接输出各阀门动作，网络内部不写任何控制公式、辨识算法或优化求解器。训练阶段完全不做模仿学习，而是直接将真实闭环控制代价穿过可微物理仿真器反向传播，并在从未见过的随机装置与机理装置上零样本考核。",
        )}
      </p>

      <div className="flex flex-col gap-4">
        <IODiagram L={L} tr={tr} />
        <NetworkDiagram />
        <TrainingDiagram />
      </div>

      <details className="group mt-4 rounded-[10px] border px-4 py-3">
        <summary className="cursor-pointer text-sm font-semibold">
          {tr("Full technical specification: random plant generator, long stays, mechanistic benchmarks, and scoring", "展开完整技术说明：随机训练场、长驻工况、30 类机理装置与打分细则")}
        </summary>
        <p className="mt-3 mb-3 max-w-[900px] text-sm leading-relaxed text-foreground/85">{L(TERMS)}</p>
        <div className="grid gap-3 lg:grid-cols-2">
          {BLOCKS.map((b, i) => <BlockCard key={i} b={b} L={L} />)}
        </div>
        <div className="mt-3"><MechCard L={L} tr={tr} /></div>
      </details>
    </Section>
  )
}

function BlockCard({ b, L }: { b: Block; L: (t: Item) => string }) {
  return (
    <article className="rounded-[10px] border bg-card p-4">
      <h4 className="mb-2 text-sm font-semibold">{L(b.h)}</h4>
      <ul className="list-disc space-y-1.5 pl-4">{b.items.map((t, i) => <li key={i} className="text-sm leading-relaxed text-foreground/80">{L(t)}</li>)}</ul>
    </article>
  )
}

function MechCard({ L, tr }: { L: (t: Item) => string; tr: (en: string, zh: string) => string }) {
  return (
    <article className="rounded-[10px] border bg-card p-4">
      <h4 className="mb-2 text-sm font-semibold">{tr("Mechanistic process benchmarks (30 plants)", "经典机理装置考试集（共 30 类）")}</h4>
      <p className="mb-2 text-sm leading-relaxed text-foreground/80">
        {tr(
          "Built from published physical and chemical engineering equations rather than the random generator. Used for zero-shot evaluation; the held-out group never enters training.",
          "依据公开化工与过程控制文献方程构建，而非随机生成器采样；用于检验模型的零样本跨装置泛化能力，其中留出组永远不进入训练。",
        )}
      </p>
      <ul className="space-y-2">{MECH.map(([h, t, c], i) => (
        <li key={i} className="border-t pt-2 text-sm first:border-t-0 first:pt-0">
          <span className="font-medium">{L(h)} <span className="num text-muted-foreground">· {c}</span></span>
          <p className="leading-relaxed text-foreground/80">{L(t)}</p>
        </li>
      ))}</ul>
    </article>
  )
}
