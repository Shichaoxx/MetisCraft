"""Geometry-only TikZ figure specifications; all prose lives in labels.json."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tensor:
    symbol: str
    shape: str
    rows: int
    cols: int
    role: str = "Hid"
    depth: int = 0
    support: str | None = None


@dataclass
class Operation:
    label: str
    width: float = 2.8
    height: float = 1.25
    literal: bool = False


@dataclass
class Operator:
    symbol: str
    width: float = 0.4
    height: float = 0.5


class Figure:
    def __init__(self, slug, formula, axes, meaning):
        self.slug, self.formula, self.axes, self.meaning = slug, formula, axes, meaning
        self.rows = []
        self.no_connections = set()

    def row(self, label, *items, connect=True):
        if not connect:
            self.no_connections.add(len(self.rows))
        self.rows.append((label, items))
        return self


def figures():
    toolkit = Figure("metiscraft-architecture", "", "", "")
    architecture = Figure(
        "jevlike-architecture",
        r"X\;\xrightarrow{\mathrm{Qwen3\ decoder}}\;H\;"
        r"\xrightarrow{P\;\mathrm{gather}}\;h\;"
        r"\xrightarrow{1024\to256\to1}\;S\;"
        r"\xrightarrow{\tau,\;\mathrm{normalize}}\;p",
        "architecture.axes",
        "architecture.meaning",
    )
    architecture.row(
        "architecture.input",
        Operation("architecture.prompt", 4.6),
        Tensor("X", "[B,T]", 2, 12, "Ctx"),
        Operation("architecture.embedding", 3.5),
        Tensor("E", "[B,T,d]", 12, 5, "Hid", 1),
    )
    architecture.row(
        "architecture.backbone",
        Tensor("E", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("architecture.block", 7.3, 2.25),
        Operation("architecture.norm", 2.6),
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
    )
    architecture.row(
        "architecture.readout",
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("slots", 2.9),
        Tensor("h", "[B,K,d]", 3, 5, "Hid", 1),
        Operation("scalarhead", 3.3),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    architecture.row(
        "architecture.output",
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("architecture.probabilities", 5.2),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
        Operation("flow.output", 3.3),
    )
    token = Figure(
        "readout-1-lm-token",
        (
            r"h_b=H_{b,t_b^\star,:},\quad z=hW_{\mathrm{lm}}^{\mathsf T},"
            r"\quad S_{b,k}=z_{b,\ell_{b,k}},\quad p_b=\operatorname{softmax}_k(S_b/\tau)"
        ),
        "token.axes",
        "token.meaning",
    )
    token.row(
        "token.stage1",
        Tensor("X", "[B,T]", 2, 12, "Ctx"),
        Operation("decoder"),
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("last"),
        Tensor("h", "[B,d]", 2, 5),
    )
    token.row(
        "token.stage2",
        Tensor("h", "[B,d]", 2, 5),
        Operator(r"\times"),
        Tensor(r"W_{\mathrm{lm}}^{\mathsf T}", "[d,V]", 5, 12, "Wgt"),
        Operator("="),
        Tensor("z", "[B,V]", 2, 12, "Lgt"),
        Operation("vocabgather"),
        Tensor("S", "[B,L]", 2, 3, "Lgt"),
    )
    listed = Figure(
        "scalar-list",
        (
            r"h_{b,k}=H_{b,P_{b,k},:},"
            r"\quad u=\operatorname{GELU}(hW_1^{\mathsf T}+b_1),"
            r"\quad S=\operatorname{squeeze}(uW_2^{\mathsf T}+b_2)"
        ),
        "list.axes",
        "list.meaning",
    )
    listed.row(
        "list.stage1",
        Tensor("X", "[B,T]", 2, 12, "Ctx"),
        Operation("decoder"),
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("slots"),
        Tensor("h", "[B,K,d]", 3, 5, "Hid", 1),
    )
    listed.row(
        "list.stage2",
        Tensor("h", "[B,K,d]", 3, 5, "Hid", 1),
        Operation("scalarhead", 3.3),
        Tensor("v", "[B,K,1]", 3, 1, "Lgt", 1),
        Operation("squeeze"),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    pair = Figure(
        "readout-3-candidate-scalar",
        (
            r"R=\sum_b K_b,\quad s_r=f_\theta(H_{r,-1,:}),"
            r"\quad S_{\operatorname{row}(r),\operatorname{col}(r)}=s_r"
        ),
        "pairs.axes",
        "pairs.meaning",
    )
    pair.row(
        "pairs.stage1",
        Tensor("X", "[R,T]", 6, 12, "Ctx"),
        Operation("decoder"),
        Tensor("H", "[R,T,d]", 12, 5, "Hid", 2),
        Operation("last"),
        Tensor("h", "[R,d]", 6, 5),
    )
    pair.row(
        "pairs.stage2",
        Tensor("h", "[R,d]", 6, 5),
        Operation("scalarhead", 3.3),
        Tensor("s", "[R]", 6, 1, "Lgt"),
        Operation("scatter", 3.3),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    flow = Figure(
        "metiscraft-tensor-flow",
        (
            r"\mathrm{Record}\;\longrightarrow\;(X,A,P)\;\longrightarrow\;H\;"
            r"\longrightarrow\;S\;\longrightarrow\;p\;\longrightarrow\;\mathrm{typed\ output}"
        ),
        "flow.axes",
        "flow.meaning",
    )
    flow.row(
        "flow.title",
        Operation("flow.input", 3.5),
        Operation("flow.pack", 3.5),
        Tensor("X", "[B,T]", 2, 12, "Ctx"),
        Operation("flow.decoder", 3.5),
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
    )
    flow.row(
        "flow.token.title",
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("flow.token", 3.6),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("flow.probs", 3.8),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
        Operation("flow.output", 3.3),
    )
    flow.row(
        "flow.scalar.title",
        Operation("flow.candidatehidden", 4.1),
        Operation("flow.scalar", 3.6),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("flow.probs", 3.8),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
        Operation("flow.output", 3.3),
    )
    notation = Figure(
        "readout-0-notation",
        (
            r"X\in\mathbb Z^{B\times T},\quad H\in\mathbb R^{B\times T\times d},"
            r"\quad S\in\mathbb R^{B\times K},\quad \tau>0"
        ),
        "notation.axes",
        "notation.meaning",
    )
    notation.row(
        "notation.stage1",
        Tensor("X", "[B,T]", 2, 12, "Ctx"),
        Tensor("A", "[B,T]", 2, 12, "Msk", support="padding"),
        Tensor("P", "[B,K]", 2, 3, "Idx"),
        Tensor(r"M^{cand}", "[B,K]", 2, 3, "Msk", support="candidate"),
        connect=False,
    )
    notation.row(
        "notation.masks",
        Tensor(r"M^{attn}", "[T,T]", 12, 12, "Msk", support="causal"),
        Operation("notation.support", 5.5),
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        connect=False,
    )
    notation.row(
        "notation.stage2",
        Tensor("h", "[B,d]", 2, 5),
        Operator(r"\times"),
        Tensor(r"W^{\mathsf T}", "[d,V]", 5, 12, "Wgt"),
        Operator("="),
        Tensor("z", "[B,V]", 2, 12, "Lgt"),
        Operation("vocabgather"),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    slot = Figure(
        "readout-2-slot-head",
        (
            r"W_{slot}^{(0)}=W_{lm}[\ell,:],\qquad z=hW_{slot}^{\mathsf T},\qquad p=\oper"
            r"atorname{softmax}_k(z/\tau)"
        ),
        "slot.axes",
        "slot.meaning",
    )
    slot.row(
        "slot.stage1",
        Tensor(r"W_{lm}", "[V,d]", 12, 5, "Wgt"),
        Operation("slot.select", 3.4),
        Tensor(r"W_{slot}", "[K,d]", 3, 5, "Wgt"),
        Operation("slot.train", 3.4),
    )
    slot.row(
        "slot.stage2",
        Tensor("h", "[B,d]", 2, 5),
        Operator(r"\times"),
        Tensor(r"W_{slot}^{\mathsf T}", "[d,K]", 5, 3, "Wgt"),
        Operator("="),
        Tensor("z", "[B,K]", 2, 3, "Lgt"),
        Operation("softmax"),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
    )
    pointer = Figure(
        "readout-4-pointer",
        (
            r"q=W_qh^{dec},\quad k_j=W_kh^{opt}_j,"
            r"\quad S_{b,j}=\frac{q_b^{\mathsf T}k_{b,j}}{\sqrt{d_p}},"
            r"\quad p=\operatorname{softmax}_j(S/\tau)"
        ),
        "pointer.axes",
        "pointer.meaning",
    )
    pointer.row(
        "pointer.stage1",
        Tensor("H", "[C,T,d]", 12, 5, "Hid"),
        Operation("pointer.gather", 4.0),
        Tensor(r"h^{dec}", "[B,d]", 2, 5),
        Operation("pointer.q", 3.2),
        Tensor("q", "[B,d_p]", 2, 4),
    )
    pointer.row(
        "pointer.stage2",
        Tensor(r"h^{opt}", "[B,K,d]", 3, 5, "Hid", 1),
        Operation("pointer.k", 3.2),
        Tensor("k", "[B,K,d_p]", 3, 4, "Hid", 1),
        Operation("pointer.dot", 4.0),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    joint = Figure(
        "readout-5-joint-schema",
        (
            r"O=\operatorname{Route}(O^0;Mem),"
            r"\quad F=\operatorname{DecoderHead}(F^0;Mem),"
            r"\quad S_{f,k}=\pi_{f,k}+\sigma(\gamma)\,g(F_f,O_k)"
        ),
        "joint.axes",
        "joint.meaning",
    )
    joint.row(
        "joint.stage1",
        Tensor("H", "[C,T,d]", 12, 5),
        Operation("joint.project", 4.2),
        Tensor("Mem", "[C,T,w]", 12, 4),
        Operation("joint.route", 4.2),
        Tensor("O", "[C,Q,K,w]", 3, 4, "Hid", 1),
    )
    joint.row(
        "joint.stage2",
        Tensor("O", "[C,Q,K,w]", 3, 4, "Hid", 1),
        Operation("joint.fields", 4.2),
        Tensor("F", "[C,Q,w]", 2, 4),
        Operation("joint.score", 4.4),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    encoder = Figure(
        "readout-6-encoder",
        (
            r"S^{uni}_{b,k}=g(H^{label}_{b,k},\operatorname{pool}(H^{text}_b)),"
            r"\quad S^{bi}=UE^{\mathsf T}/\tau,\quad S^{cross}_r=f(Encoder(x_b,c_k))"
        ),
        "encoder.axes",
        "encoder.meaning",
    )
    encoder.row(
        "encoder.stage1",
        Tensor("H", "[B,T,d]", 12, 5, "Hid", 1),
        Operation("encoder.pool", 4.1),
        Tensor("u", "[B,d]", 2, 5),
        Operator(";"),
        Tensor("E", "[B,K,d]", 3, 5, "Hid", 1),
        Operation("encoder.compare", 3.6),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
    )
    encoder.row(
        "encoder.stage2",
        Tensor("U", "[B,d]", 2, 5),
        Operator(r"\times"),
        Tensor(r"E^{\mathsf T}", "[d,K]", 5, 3),
        Operator("="),
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("encoder.cache", 4.1),
        connect=False,
    )
    encoder.row(
        "encoder.stage3",
        Tensor("X_{pair}", "[R,T]", 6, 12, "Ctx"),
        Operation("encoder.cross", 4.1),
        Tensor("h", "[R,d]", 6, 5),
        Operation("scalarhead", 3.3),
        Tensor("s", "[R]", 6, 1, "Lgt"),
    )
    generated = Figure(
        "readout-7-generate-then-read",
        (
            r"r_{1:G}\sim\pi_\theta(\cdot\mid x),\quad X'=[x;r;\mathrm{answer\ cue}],"
            r"\quad p=\operatorname{softmax}_k(z_{\ell_k}/\tau)"
        ),
        "generated.axes",
        "generated.meaning",
    )
    generated.row(
        "generated.stage1",
        Operation("generated.input", 3.8),
        Operation("generated.decode", 4.1),
        Operation("generated.concat", 4.2),
        Tensor("X'", "[B,T]", 2, 12, "Ctx"),
    )
    generated.row(
        "generated.stage2",
        Tensor("X'", "[B,T]", 2, 12, "Ctx"),
        Operation("decoder"),
        Tensor("h", "[B,d]", 2, 5),
        Operation("generated.token", 4.5),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
    )
    contract = Figure(
        "system-one-contract",
        (
            r"\mathrm{state}+\{\mathrm{typed\ questions}\}\longrightarrow\{"
            r"\mathrm{decisions},\mathrm{probabilities}\}"
        ),
        "contract.axes",
        "contract.meaning",
    )
    contract.row(
        "contract.stage1",
        Operation("contract.state", 5.2),
        Operation("contract.questions", 5.2),
        Operation("contract.atomic", 5.2),
    )
    contract.row(
        "contract.stage2",
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("contract.normalize", 4.8),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
        Operation("contract.output", 5.1),
    )
    training = Figure(
        "training-calibration",
        (
            r"\mathcal L=\frac{1}{N}\sum_b\ell_b,\quad \tau=\exp(a),"
            r"\quad p=\operatorname{normalize}((S-\Delta)/\tau)"
        ),
        "training.axes",
        "training.meaning",
    )
    training.row(
        "training.stage1",
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("training.loss", 4.4),
        Operation("training.update", 3.7),
        Operation("training.split", 4.1),
    )
    training.row(
        "training.stage2",
        Tensor("Z_{completion}", "[B,2,V]", 2, 12, "Lgt", 1),
        Operator(";"),
        Tensor("y_{code,EOS}", "[B,2]", 2, 2, "Idx"),
        Operation("training.completion", 4.6),
        Tensor(r"\ell", "[B]", 2, 1, "Lgt"),
    )
    training.row(
        "training.stage3",
        Tensor("S", "[B,K]", 2, 3, "Lgt"),
        Operation("training.shift", 4.2),
        Operation("training.temperature", 4.4),
        Operation("training.normalize", 4.2),
        Tensor("p", "[B,K]", 2, 3, "Lgt"),
    )
    overview = Figure(
        "readout-overview",
        (
            r"\mathrm{input\ layout}\quad+\quad\mathrm{information\ flow}\quad+\quad"
            r"\mathrm{score\ operator}\quad\longrightarrow\quad\mathrm{typed\ judgment}"
        ),
        "overview.axes",
        "overview.meaning",
    )
    overview.row(
        "overview.stage1",
        Operation("overview.token", 5.8),
        Operation("overview.slot", 5.8),
        Operation("overview.scalar", 5.8),
        connect=False,
    )
    overview.row(
        "overview.stage2",
        Operation("overview.pointer", 5.8),
        Operation("overview.joint", 5.8),
        Operation("overview.encoder", 5.8),
        connect=False,
    )
    overview.row(
        "overview.stage3",
        Operation("overview.generated", 5.8),
        Operation("overview.list", 5.8),
        Operation("overview.scope", 5.8),
        connect=False,
    )
    return {
        f.slug: f
        for f in (
            toolkit,
            architecture,
            notation,
            token,
            slot,
            pair,
            pointer,
            joint,
            encoder,
            generated,
            overview,
            contract,
            flow,
            listed,
            training,
        )
    }
