import { useLang } from "@/lib/i18n"

/* Architecture diagrams drawn directly from live code on main:
   - NetworkDiagram: aio/seqnet.py (SeqAIO v4: width 768, 12 layers, 12 heads, QK-Norm, RoPE, dense multi-lag
     cross-channel attention across 1/3/8/16/24/36/48/64 steps, 64-dim multi-timescale token EMA memory fed back
     into Layer 0, gain-aware multi-scale Erlang-8 causal SSM + Gated DeltaNet pairwise associative memory with
     8-scale slew-normalized in-flight pipe displacement (Neural Smith Predictor) injected into Layer 0, all 12
     layers' channel & lag attention, and 1-hop edge-conditioned message passing, plus width-scaled action head
     with multiplicative log-gain precision gate; ~128.6M active runtime parameters, strictly one pure neural network).
   - TrainingDiagram: aio/train.py + aio/net.py (Privileged Critic: 64-step truncated BPTT through the differentiable
     simulator on pure real control cost with reversal-only anti-chatter smoothing, per-plant gradient balancing,
     sensitivity-scaled exploration noise, 4-head 64–256-step dead-time Smith-predictor privileged critic at the
     window boundary, training-only 75-param unpublished-pair linear probe for presence/sign/gain, hybrid Muon + AdamW + EMA). */

type Item = [string, string]

function Pill({ tone, children }: { tone: "deploy" | "train" | "feedback" | "grad"; children: React.ReactNode }) {
  const cls =
    tone === "deploy"
      ? "border-[var(--fx-black)] bg-[var(--fx-black)] text-white"
      : tone === "feedback"
        ? "border-[var(--chart-3)]/40 bg-[var(--ok-bg)] text-[var(--ok)]"
        : tone === "grad"
          ? "border-[var(--chart-2)]/40 bg-[var(--warn-bg)] text-[var(--warn)]"
          : "border-[var(--chart-1)]/35 bg-[var(--surface)] text-[var(--chart-1)]"
  return <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[13px] font-medium ${cls}`}>{children}</span>
}

function NetworkSvg() {
  const { tr } = useLang()
  return (
    <div className="overflow-x-auto rounded-[8px] border bg-[var(--surface)] p-2 sm:p-3">
      <svg
        viewBox="0 0 1020 442"
        className="mx-auto block h-auto min-w-[760px] w-full"
        role="img"
        aria-label={tr("End-to-end control network architecture diagram", "端到端控制网络结构示意图")}
      >
        <defs>
          <marker id="arr-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--fx-black)" />
          </marker>
          <marker id="arr-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--chart-3)" />
          </marker>
          <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--chart-1)" />
          </marker>
        </defs>

        {/* Outer deployment boundary */}
        <rect x="8" y="8" width="1004" height="426" rx="10" fill="#ffffff" stroke="var(--fx-black)" strokeWidth="1.5" />
        <rect x="20" y="18" width="320" height="24" rx="12" fill="var(--fx-black)" />
        <text x="180" y="34" textAnchor="middle" fontSize="12" fontWeight="600" fill="#ffffff">
          {tr("Deployed Runtime · One Pure Neural Network", "部署运行态 · 单一纯神经网络（无公式与求解器）")}
        </text>
        <text x="996" y="34" textAnchor="end" fontSize="12" fill="var(--fx-black-4)">
          {tr("~128M active params · shared weights across any plant size", "约 1.28 亿活跃参数 · 共享权重原生适配任意装置规模")}
        </text>

        {/* Column 1: Inputs & Tokenization */}
        <rect x="22" y="64" width="186" height="84" rx="7" fill="var(--surface)" stroke="var(--border)" strokeWidth="1.4" />
        <text x="115" y="88" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--fx-black)">
          {tr("CV & DV Readings", "被控量与可测扰动")}
        </text>
        <text x="115" y="108" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("12 dimensionless features", "每路 12 维无量纲特征")}
        </text>
        <text x="115" y="128" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Shared Encoder → 768d", "共享线性编码 → 768 维")}
        </text>

        <rect x="22" y="162" width="186" height="84" rx="7" fill="var(--surface)" stroke="var(--border)" strokeWidth="1.4" />
        <text x="115" y="186" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--fx-black)">
          {tr("Valve States (MV)", "阀门状态与上步动作")}
        </text>
        <text x="115" y="206" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("6 features + last move", "6 维状态 + 上一步增量")}
        </text>
        <text x="115" y="226" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Shared Encoder → 768d", "共享线性编码 → 768 维")}
        </text>

        {/* Optional pairing hint box */}
        <rect x="22" y="262" width="186" height="50" rx="7" fill="#ffffff" stroke="var(--chart-1)" strokeWidth="1.3" strokeDasharray="4 3" />
        <text x="115" y="283" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--chart-1)">
          {tr("Loop Pairing (optional)", "回路配对提示（可选）")}
        </text>
        <text x="115" y="301" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Learned attention bias only", "仅作可学习注意力偏置")}
        </text>

        {/* Layer 0 Sum node */}
        <circle cx="246" cy="155" r="15" fill="var(--ok-bg)" stroke="var(--chart-3)" strokeWidth="1.8" />
        <text x="246" y="160" textAnchor="middle" fontSize="16" fontWeight="700" fill="var(--ok)">+</text>
        <line x1="208" y1="106" x2="233" y2="145" stroke="var(--fx-black)" strokeWidth="1.6" markerEnd="url(#arr-ink)" />
        <line x1="208" y1="204" x2="233" y2="165" stroke="var(--fx-black)" strokeWidth="1.6" markerEnd="url(#arr-ink)" />
        <line x1="261" y1="155" x2="288" y2="155" stroke="var(--fx-black)" strokeWidth="1.8" markerEnd="url(#arr-ink)" />

        {/* Pairing path into Backbone C */}
        <path d="M208 287 H268 V246 H288" fill="none" stroke="var(--chart-1)" strokeWidth="1.4" strokeDasharray="4 3" markerEnd="url(#arr-blue)" />

        {/* Column 2: 12-Layer Spatio-Temporal Backbone */}
        <rect x="292" y="60" width="326" height="248" rx="9" fill="var(--fx-black)" />
        <text x="455" y="84" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="#ffffff">
          {tr("12-Layer Attention Backbone (×12)", "12 层时空交替注意力主干（重复 ×12）")}
        </text>
        <text x="455" y="102" textAnchor="middle" fontSize="12" fill="var(--fx-lime)">
          {tr("Width 768 · 12 heads · Pre-LayerNorm + Residual", "宽 768 · 12 注意力头 · 预归一化残差连接")}
        </text>

        {/* 4 sub-blocks inside the backbone */}
        <rect x="308" y="114" width="294" height="39" rx="5" fill="#18202c" stroke="#364152" strokeWidth="1.2" />
        <text x="320" y="131" fontSize="12.5" fontWeight="600" fill="#ffffff">{tr("A. Causal Temporal Attention", "A. 因果时间注意力（自身过去 128 步）")}</text>
        <text x="320" y="147" fontSize="11.5" fill="#cbd5e1">{tr("Past 128 steps · RoPE · QK-Norm · KV Cache", "旋转位置编码 · QK 归一化 · 推理复用键值缓存")}</text>

        <rect x="308" y="161" width="294" height="39" rx="5" fill="#18202c" stroke="var(--chart-3)" strokeWidth="1.2" />
        <text x="320" y="178" fontSize="12.5" fontWeight="600" fill="var(--fx-lime)">{tr("B. Multi-Lag Cross-Channel Attention", "B. 多滞后跨通道注意力（直连死时间）")}</text>
        <text x="320" y="194" fontSize="11.5" fill="#cbd5e1">{tr("Lags 1,3,8,16,24,36,48,64 + DeltaNet bias", "回看 1–64 步 8 档延迟 + DeltaNet 滞后偏置")}</text>

        <rect x="308" y="208" width="294" height="39" rx="5" fill="#18202c" stroke="var(--chart-3)" strokeWidth="1.3" />
        <text x="320" y="225" fontSize="12.5" fontWeight="600" fill="#ffffff">{tr("C. Cross-Channel Interaction Attention", "C. 同刻跨通道交互注意力（解耦与让步）")}</text>
        <text x="320" y="241" fontSize="11.5" fill="#cbd5e1">{tr("All CVs/MVs + pairing bias + DeltaNet bias", "全体通道交互 + 配对偏置 + DeltaNet 耦合偏置")}</text>

        <rect x="308" y="255" width="294" height="39" rx="5" fill="#18202c" stroke="#364152" strokeWidth="1.2" />
        <text x="320" y="272" fontSize="12.5" fontWeight="600" fill="#ffffff">{tr("D. Feed-Forward Network (FFN)", "D. 逐通道前馈网络（FFN）")}</text>
        <text x="320" y="288" fontSize="11.5" fill="#cbd5e1">{tr("Hidden width 3,072 · SiLU activation", "隐层宽度 3,072 · SiLU 非线性激活")}</text>

        {/* Backbone output arrow */}
        <line x1="618" y1="155" x2="656" y2="155" stroke="var(--fx-black)" strokeWidth="1.8" markerEnd="url(#arr-ink)" />

        {/* Column 3: Action Head & Two Persistent Memory Boxes (clean right corridor at x=990) */}
        <rect x="660" y="64" width="308" height="106" rx="8" fill="#ffffff" stroke="var(--fx-black)" strokeWidth="2" />
        <text x="814" y="88" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--fx-black)">
          {tr("Direct Action Readout Head", "直接动作输出头（每阀门独立读出）")}
        </text>
        <text x="814" y="109" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("MV token + long memory + 1-hop decoupling", "阀门特征 + 长时记忆 + 1 跳多变量解耦消息")}
        </text>
        <text x="814" y="128" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Width scale × precision gate → tanh bound", "宽度缩放 × 微幅精调门控 → tanh 速率限幅")}
        </text>
        <rect x="678" y="139" width="272" height="22" rx="4" fill="var(--fx-black)" />
        <text x="814" y="154" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--fx-lime)">
          {tr("Output: Direct Valve Move Increment Δu_t", "输出：直接阀门动作增量 Δu_t")}
        </text>

        <rect x="660" y="190" width="308" height="66" rx="7" fill="var(--ok-bg)" stroke="var(--chart-3)" strokeWidth="1.5" />
        <text x="814" y="212" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--ok)">
          {tr("Per-Channel Long Memory (64-dim EMA)", "通道长时记忆（64 维多尺度指数移动平均）")}
        </text>
        <text x="814" y="231" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Timescales 4–2,048 steps · adds to output", "遗忘周期 4–2,048 步 · 捕捉慢漂移与液位趋势")}
        </text>
        <text x="814" y="247" textAnchor="middle" fontSize="12" fill="var(--ok)">
          {tr("Feeds back into Layer 0 on next step", "下一步经右侧回路直接回馈到第 0 层")}
        </text>

        <rect x="660" y="272" width="308" height="66" rx="7" fill="var(--ok-bg)" stroke="var(--chart-3)" strokeWidth="1.5" />
        <text x="814" y="294" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--ok)">
          {tr("Gain-Aware DeltaNet + Pipe Memory", "增益感知 DeltaNet 与在途管道记忆")}
        </text>
        <text x="814" y="313" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("8-stage SSM + Delta rule + gain & pipe lag", "8 阶状态空间 + Delta 规则 + 增益与在途位移")}
        </text>
        <text x="814" y="329" textAnchor="middle" fontSize="12" fill="var(--ok)">
          {tr("Layer-0 Smith pred + 12 layers + 1-hop head", "第 0 层 Smith 预测 + 12 层偏置 + 1 跳解耦")}
        </text>

        {/* Split down from backbone output to memory boxes */}
        <path d="M638 155 V223 H656" fill="none" stroke="var(--fx-black)" strokeWidth="1.6" markerEnd="url(#arr-ink)" />
        <path d="M638 223 V305 H656" fill="none" stroke="var(--fx-black)" strokeWidth="1.6" markerEnd="url(#arr-ink)" />
        {/* Memory adds back to readout */}
        <line x1="814" y1="190" x2="814" y2="174" stroke="var(--chart-3)" strokeWidth="1.6" markerEnd="url(#arr-green)" />

        {/* Feedback loop 1: routes OUTSIDE the right edge of memory boxes -> bottom channel -> Layer 0 (+) with ZERO box overlap */}
        <path
          d="M968 223 H992 V408 H246 V174"
          fill="none"
          stroke="var(--chart-3)"
          strokeWidth="1.8"
          strokeDasharray="6 4"
          markerEnd="url(#arr-green)"
        />
        <rect x="330" y="395" width="320" height="24" rx="5" fill="#ffffff" stroke="var(--chart-3)" strokeWidth="1.1" />
        <text x="490" y="411" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--ok)">
          {tr("Next-step feedback ①: Long memory → Layer 0", "跨步反馈 ①：通道长时记忆 → 注入第 0 层")}
        </text>

        {/* Feedback loop 2: routes cleanly from bottom of Pairwise box -> y=366 -> Backbone B/C */}
        <path
          d="M814 338 V368 H455 V312"
          fill="none"
          stroke="var(--chart-3)"
          strokeWidth="1.8"
          strokeDasharray="6 4"
          markerEnd="url(#arr-green)"
        />
        <rect x="484" y="355" width="352" height="24" rx="5" fill="#ffffff" stroke="var(--chart-3)" strokeWidth="1.1" />
        <text x="660" y="371" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--ok)">
          {tr("Feedback ②: DeltaNet memory → 12 layers (B/C) + Head", "跨步反馈 ②：DeltaNet 记忆 → 12 层（B/C）与动作头")}
        </text>
      </svg>
    </div>
  )
}

function TrainingSvg() {
  const { tr } = useLang()
  return (
    <div className="overflow-x-auto rounded-[8px] border bg-[var(--surface)] p-2 sm:p-3">
      <svg
        viewBox="0 0 1040 462"
        className="mx-auto block h-auto min-w-[780px] w-full"
        role="img"
        aria-label={tr("End-to-end differentiable training architecture diagram", "可微仿真端到端训练架构示意图")}
      >
        <defs>
          <marker id="t-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--fx-black)" />
          </marker>
          <marker id="t-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--chart-1)" />
          </marker>
          <marker id="t-orange" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M1 1.5 L8.5 5 L1 8.5 Z" fill="var(--chart-2)" />
          </marker>
        </defs>

        {/* Left: Deployed Control Network */}
        <rect x="14" y="18" width="242" height="222" rx="10" fill="var(--fx-black)" />
        <rect x="28" y="32" width="184" height="22" rx="11" fill="var(--fx-lime)" />
        <text x="120" y="47" textAnchor="middle" fontSize="12" fontWeight="700" fill="var(--fx-black)">
          {tr("Deployed Policy Network", "唯一部署 · 控制网络")}
        </text>
        <text x="135" y="80" textAnchor="middle" fontSize="14" fontWeight="600" fill="#ffffff">
          {tr("End-to-End Model", "端到端控制网络")}
        </text>
        <text x="135" y="100" textAnchor="middle" fontSize="12" fill="var(--fx-lime)">
          {tr("~128M active parameters", "约 1.28 亿活跃参数")}
        </text>
        <text x="135" y="128" textAnchor="middle" fontSize="12" fill="#e2e8f0">
          {tr("Sees ONLY noisy public", "仅看公开无量纲含噪仪表")}
        </text>
        <text x="135" y="146" textAnchor="middle" fontSize="12" fill="#e2e8f0">
          {tr("instrument observations", "不接触方程与内部真值")}
        </text>
        <rect x="30" y="170" width="210" height="54" rx="6" fill="#18202c" stroke="#364152" />
        <text x="135" y="191" textAnchor="middle" fontSize="12" fontWeight="600" fill="#ffffff">
          {tr("EMA Weight Shadow (0.97)", "指数移动平均权重（EMA 0.97）")}
        </text>
        <text x="135" y="210" textAnchor="middle" fontSize="11.5" fill="#cbd5e1">
          {tr("Evaluated every 10 rounds", "每 10 轮评测并保存最优权重")}
        </text>

        {/* Right big dashed box: Training-Only Infrastructure */}
        <rect x="322" y="10" width="704" height="320" rx="10" fill="#ffffff" stroke="var(--chart-1)" strokeWidth="1.8" strokeDasharray="7 5" />
        <rect x="338" y="20" width="440" height="24" rx="12" fill="var(--surface)" stroke="var(--chart-1)" strokeWidth="1.2" />
        <text x="558" y="36" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--chart-1)">
          {tr("Training-Only Infrastructure (removed at deployment)", "仅训练期组件（上线部署时全部移除）")}
        </text>

        {/* Closed-loop arrows between Policy & Differentiable Simulator */}
        <line x1="256" y1="88" x2="338" y2="88" stroke="var(--fx-black)" strokeWidth="1.8" markerEnd="url(#t-ink)" />
        <text x="297" y="80" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--fx-black)">{tr("u_t", "阀门 u_t")}</text>
        <line x1="342" y1="128" x2="260" y2="128" stroke="var(--fx-black)" strokeWidth="1.8" markerEnd="url(#t-ink)" />
        <text x="297" y="146" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--fx-black)">{tr("o_t", "观测 o_t")}</text>
        <rect x="268" y="160" width="52" height="24" rx="4" fill="var(--warn-bg)" stroke="var(--chart-2)" />
        <text x="294" y="176" textAnchor="middle" fontSize="11.5" fontWeight="600" fill="var(--warn)">{tr("×64", "64 步")}</text>

        {/* Simulator box (widened to 316px so every line has generous margins) */}
        <rect x="342" y="56" width="316" height="128" rx="8" fill="var(--surface)" stroke="var(--chart-1)" strokeWidth="1.5" />
        <text x="500" y="79" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--fx-black)">
          {tr("Differentiable Plant Simulator", "多规模可微物理仿真器")}
        </text>
        <text x="500" y="100" textAnchor="middle" fontSize="12" fill="var(--chart-1)">
          {tr("Batches: 12×5 (80) · 24×10 (28) · 48×20 (18)", "并行批次：12×5 (80) · 24×10 (28) · 48×20 (18)")}
        </text>
        <text x="500" y="121" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Dead time, coupling, nonlinearity, levels, faults", "含纯滞后、强耦合、非线性、液位积分与故障")}
        </text>
        <text x="500" y="140" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("75% paired · 25% hidden pairing (probing)", "75% 告知配对 · 25% 隐藏配对（允许试阀）")}
        </text>
        <text x="500" y="162" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--fx-black)">
          {tr("64-Step Differentiable Closed-Loop Rollout", "64 步闭环截断反向传播窗口（Truncated BPTT）")}
        </text>

        {/* Right top: Real Closed-Loop Cost */}
        <rect x="690" y="56" width="312" height="114" rx="8" fill="#ffffff" stroke="var(--fx-black)" strokeWidth="1.5" />
        <text x="846" y="79" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--fx-black)">
          {tr("Real Control Cost + Training-Only Probe", "真实闭环代价 + 仅训练期配对探针")}
        </text>
        <text x="846" y="100" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Zero imitation · Tracking + limits + trips", "零模仿学习 · 超带宽 + 区间越限 + 失稳跳停")}
        </text>
        <text x="846" y="119" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("+ Reversal-only smooth + 0.5 floor balance", "+ 反向防抖平滑 + 0.5 下限单样本梯度平衡")}
        </text>
        <text x="846" y="138" textAnchor="middle" fontSize="12" fill="var(--ok)">
          {tr("+ 75-param unpublished-pair probe (train only)", "+ 75 参数未告知配对探针（仅训练期，零部署开销）")}
        </text>
        <text x="846" y="157" textAnchor="middle" fontSize="11.5" fill="var(--fx-black-4)">
          {tr("Calibrates presence, sign & physical gain scale", "校准存在性、正反作用与物理增益量级")}
        </text>

        {/* Arrow Simulator -> Real Cost */}
        <line x1="658" y1="108" x2="686" y2="108" stroke="var(--fx-black)" strokeWidth="1.7" markerEnd="url(#t-ink)" />

        {/* Right bottom: Privileged Critic */}
        <rect x="690" y="186" width="312" height="128" rx="8" fill="#f0f7ff" stroke="var(--chart-1)" strokeWidth="1.6" />
        <text x="846" y="208" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--chart-1)">
          {tr("Dead-Time Privileged Critic (~2M)", "纯滞后特权价值网络（约 200 万参数）")}
        </text>
        <text x="846" y="228" textAnchor="middle" fontSize="12" fill="var(--fx-black)">
          {tr("Reads exact noiseless state at step 64:", "在窗口末端读取仿真器无噪声内部真值：")}
        </text>
        <text x="846" y="247" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Delay-pipe moves · Smith error · level drift", "延迟管道在途动作 · Smith 终态误差 · 液位漂移")}
        </text>
        <text x="846" y="266" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("4 fixed-horizon heads (64–256 future steps)", "4 个定长远期头（覆盖未来 64–256 步）")}
        </text>
        <text x="846" y="286" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--chart-1)">
          {tr("Fitted on realized future costs (no TD drift)", "用后续真实回放代价监督拟合（零自举发散）")}
        </text>
        <text x="846" y="304" textAnchor="middle" fontSize="11.5" fill="var(--fx-black-4)">
          {tr("Gives differentiable terminal gradient past step 64", "提供 64 步之后的平滑远期梯度")}
        </text>

        {/* Arrow Simulator step 64 true state -> Privileged Critic */}
        <path d="M500 184 V248 H686" fill="none" stroke="var(--chart-1)" strokeWidth="1.7" markerEnd="url(#t-blue)" />
        <rect x="518" y="228" width="150" height="36" rx="5" fill="#ffffff" stroke="var(--chart-1)" strokeWidth="1" />
        <text x="593" y="243" textAnchor="middle" fontSize="11.5" fontWeight="600" fill="var(--chart-1)">{tr("True state s_64", "第 64 步真值状态")}</text>
        <text x="593" y="258" textAnchor="middle" fontSize="11" fill="var(--fx-black-4)">{tr("(differentiable)", "（保留求导图）")}</text>

        {/* Backprop gradient flow: clean right corridor x=1016 -> horizontal track y=406 -> Optimizer box right edge x=480 */}
        <path
          d="M1002 114 H1016 V406 H482"
          fill="none"
          stroke="var(--chart-2)"
          strokeWidth="2.1"
          strokeDasharray="6 4"
          markerEnd="url(#t-orange)"
        />
        <line x1="1002" y1="250" x2="1016" y2="250" stroke="var(--chart-2)" strokeWidth="2.1" strokeDasharray="6 4" />

        {/* Gradient flow label pill seated ABOVE the horizontal dashed arrow (zero overlap with arrows!) */}
        <rect x="504" y="356" width="496" height="32" rx="6" fill="#ffffff" stroke="var(--chart-2)" strokeWidth="1.3" />
        <text x="752" y="376" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--warn)">
          {tr("Truncated BPTT: (64-Step Cost + Critic Value) → Backprop via Simulator", "端到端梯度反传：64 步真实代价 + 特权远期价值 → 穿越仿真器求导")}
        </text>

        {/* Optimizer + Adaptive Guard Box at bottom left (generous 462px width!) */}
        <rect x="14" y="352" width="462" height="94" rx="8" fill="var(--warn-bg)" stroke="var(--chart-2)" strokeWidth="1.6" />
        <text x="245" y="376" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--warn)">
          {tr("Adaptive Guard + Muon & AdamW Optimizer", "自适应梯度护栏 + Muon / AdamW 混合优化器")}
        </text>
        <text x="245" y="398" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("Per-plant spike guard · Muon (2D matrices) + AdamW (biases)", "单台尖峰护栏 · 主干二维矩阵用 Muon · 偏置/归一化用 AdamW")}
        </text>
        <text x="245" y="418" textAnchor="middle" fontSize="12" fill="var(--fx-black-4)">
          {tr("4× routing-bias LR · Cosine decay · Updates policy & EMA", "路由偏置 4 倍学习率 · 余弦退火 · 同步更新网络与 EMA 权重")}
        </text>

        {/* Optimizer arrow up into Deployed Policy Network */}
        <line x1="135" y1="352" x2="135" y2="245" stroke="var(--chart-2)" strokeWidth="2.1" strokeDasharray="6 4" markerEnd="url(#t-orange)" />
        <rect x="26" y="280" width="218" height="32" rx="5" fill="#ffffff" stroke="var(--chart-2)" strokeWidth="1.1" />
        <text x="135" y="300" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--warn)">
          {tr("Parameter update → Control Policy", "梯度更新 → 控制网络参数")}
        </text>
      </svg>
    </div>
  )
}

const NET_NOTES: { h: Item; t: Item }[] = [
  {
    h: ["Why variable-sized plants share one network", "为什么一套权重能控任意规模装置"],
    t: [
      "Every controlled variable, disturbance, and valve uses the same shared encoder and attention layers; nothing in the weights depends on loop count. A model trained on 12×5, 24×10, and 48×20 handles unseen sizes natively.",
      "每个被控量、扰动和阀门都走共享编码器与注意力层，权重维度与装置变量个数无关，因此同一模型无需改结构就能直接控制不同规模的装置。",
    ],
  },
  {
    h: ["Why multi-lag attention matters for dead time", "为什么要加 8 档多滞后跨通道注意力"],
    t: [
      "In industrial processes, a valve move may take 20–60 steps to reach a downstream sensor. Direct cross-channel attention at lags 1, 3, 8, 16, 24, 36, 48, and 64 lets each layer link cause and delayed effect in one step.",
      "化工与过程对象常存在几十步纯滞后（死时间）。直接跨越 1 至 64 步的 8 个延迟档位回看所有通道，使网络一步就能把历史动作和此刻才到的读数对齐。",
    ],
  },
  {
    h: ["Why gain-aware DeltaNet & Neural Smith Predictor", "为什么引入增益感知 DeltaNet 与在途管道预测"],
    t: [
      "Unpublished cross-couplings carry 65.5% of total plant gain, and extreme high-gain 55-step dead-time loops need tiny, 0.4%-of-rate moves without over-integrating during silence. An 8-stage causal state-space + Gated DeltaNet tracks multi-lag causality and physical gain sensitivity online, feeds 8-scale in-flight pipe displacement into Layer 0 (a Neural Smith Predictor), and pairs with a multiplicative precision gate spanning three orders of magnitude.",
      "未告知跨通道耦合占全装置总增益的 65.5%，而超高增益 55 步纯滞后回路需要在静默期克制积分并输出速率上限 0.4% 的极微小步幅。8 阶因果状态空间配合 Gated DeltaNet 在线追踪多滞后因果与物理增益量级，把 8 尺度在途管道位移注入第 0 层构成神经网络版 Smith 预测器，并配合跨三个数量级的微幅精调门控输出精准阀位增量。",
    ],
  },
]

const TRAIN_NOTES: { h: Item; t: Item }[] = [
  {
    h: ["Why real cost + training-only unpublished-pair probe", "为什么以真实代价为主、辅以仅训练期配对探针"],
    t: [
      "Cloning an MPC teacher breaks on blind or drifting plants, so control moves are optimized purely via real closed-loop cost through the differentiable simulator with sensitivity-scaled exploration noise. A tiny 75-parameter linear probe (removed at deployment) supervises the DeltaNet memory strictly on unpublished & hidden pairs for presence, sign, and physical gain scale.",
      "模仿某一套控制器的动作容易在盲控或漂移工况失灵，因此阀门动作完全由穿过可微仿真器的真实闭环代价配合灵敏度自适应探索噪声优化。另附一个仅训练期使用的 75 参数线性探针（部署时移除），严格在未告知与盲控配对上监督存在性、正反作用与物理增益量级。",
    ],
  },
  {
    h: ["Why use a training-only dead-time privileged critic", "为什么训练期引入纯滞后特权价值网络"],
    t: [
      "A 64-step backpropagation window alone cannot see the outcome of valve moves still traveling through a 40-step dead time or slow tank level. The privileged critic sees the exact simulator delay pipelines and predicts realized future costs over 64–256 steps—giving accurate early-braking gradients during training while adding zero overhead at deployment.",
      "仅靠 64 步反传窗口看不到“还在 40 步死时间管道里传输的动作”以及慢液位的远期后果。仅训练期使用的特权价值网络直接读取仿真内部延迟管道，用未来真实代价预测 64–256 步远期损失，给控制网络提供精准的提前收阀梯度，部署时则完全拿掉。",
    ],
  },
  {
    h: ["Why reversal-only smoothing & gradient balancing", "为什么采用反向平滑与单样本梯度平衡"],
    t: [
      "Heavy uniform move penalties make controllers sluggish; penalizing valve direction reversals specifically stops high-frequency chatter without slowing down real setpoint tracking. Balancing per-plant loss with a 0.5 floor stops hard long-delay plants from having their gradients drowned out by easy ones.",
      "粗暴增大全局动作惩罚会把控制器压得反应迟钝；只对阀门来回换向加惩罚，既能消除抖动又不影响正常纠偏速度。带 0.5 下限的单样本归一化则保证高难度大滞后装置不被简单装置稀释掉梯度。",
    ],
  },
]

export function NetworkDiagram() {
  const { tr } = useLang()
  const L = (t: Item) => tr(t[0], t[1])

  return (
    <figure className="rounded-[10px] border bg-card p-4 sm:p-5">
      <figcaption className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-base font-semibold">{tr("Network architecture: one end-to-end control model", "网络结构图：单一端到端控制网络")}</h4>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {tr(
              "A single ~128M-parameter spatio-temporal attention network with dense dead-time lag lookback and two persistent feedback memories. No formulas, identifiers, or optimization solvers exist inside.",
              "单一约 1.28 亿活跃参数的时空注意力网络，具备致密多滞后回看与两条跨步反馈记忆。网络内部不含任何控制公式、辨识器或优化求解器。",
            )}
          </p>
        </div>
        <Pill tone="deploy">{tr("Deployed standalone", "上线部署唯一模块")}</Pill>
      </figcaption>
      <NetworkSvg />
      <div className="mt-3 grid gap-2.5 md:grid-cols-3">
        {NET_NOTES.map((n, i) => (
          <div key={i} className="rounded-[8px] border bg-surface p-3">
            <p className="text-sm font-semibold">{L(n.h)}</p>
            <p className="mt-1 text-[13px] leading-relaxed text-foreground/80">{L(n.t)}</p>
          </div>
        ))}
      </div>
    </figure>
  )
}

export function TrainingDiagram() {
  const { tr } = useLang()
  const L = (t: Item) => tr(t[0], t[1])

  return (
    <figure className="rounded-[10px] border bg-card p-4 sm:p-5">
      <figcaption className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-base font-semibold">{tr("Training architecture: differentiable simulation with privileged value guidance", "训练结构图：可微仿真反传 + 特权价值引导")}</h4>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {tr(
              "64-step truncated backpropagation through time on pure real control cost, paired with a training-only 4-head dead-time privileged critic (64–256 steps) and hybrid Muon + AdamW optimization.",
              "在可微物理仿真器上做 64 步闭环截断反向传播，只优化真实控制代价；窗口末端接入仅训练期使用的四头纯滞后特权价值网络（覆盖 64–256 步），配合 Muon + AdamW 混合优化器训练。",
            )}
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Pill tone="train">{tr("Dashed box: training only", "虚线框：仅训练期使用")}</Pill>
          <Pill tone="grad">{tr("Dashed orange: gradient flow", "橙色虚线：反向传播梯度")}</Pill>
        </div>
      </figcaption>
      <TrainingSvg />
      <div className="mt-3 grid gap-2.5 md:grid-cols-3">
        {TRAIN_NOTES.map((n, i) => (
          <div key={i} className="rounded-[8px] border bg-surface p-3">
            <p className="text-sm font-semibold">{L(n.h)}</p>
            <p className="mt-1 text-[13px] leading-relaxed text-foreground/80">{L(n.t)}</p>
          </div>
        ))}
      </div>
    </figure>
  )
}

export function SeqDiagram() {
  return (
    <div className="flex flex-col gap-4">
      <NetworkDiagram />
      <TrainingDiagram />
    </div>
  )
}
