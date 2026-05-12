import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns

from codon.utils.tokens import PackedTokenizer, create_tokenizer_trainer, chat_template
from codon.utils.session import Session


console = Console()

STYLE_KEEP = "bold green"
STYLE_MASK = "grey42"


# ---------- tokenizer ----------
def build_tmp_tokenizer() -> PackedTokenizer:
    return PackedTokenizer('./motif_vocab.zip')


def load_tokenizer() -> PackedTokenizer:
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        console.print(f"[cyan]loading tokenizer:[/] {path}")
        return PackedTokenizer(path)
    console.print("[yellow]no tokenizer path given, training a mini tokenizer...[/]")
    return build_tmp_tokenizer()


# ---------- rendering ----------
def _segment_by_mask(ids: list[int], mask: list[bool]) -> list[tuple[bool, list[int]]]:
    """把 (ids, mask) 切成连续 mask 相同的段。"""
    if not ids:
        return []
    segs: list[tuple[bool, list[int]]] = []
    cur_m = mask[0]
    cur_ids = [ids[0]]
    for tid, ig in zip(ids[1:], mask[1:]):
        if ig == cur_m:
            cur_ids.append(tid)
        else:
            segs.append((cur_m, cur_ids))
            cur_m, cur_ids = ig, [tid]
    segs.append((cur_m, cur_ids))
    return segs


def render_message(msg, tokenizer: PackedTokenizer) -> Text:
    text = Text()
    for is_masked, ids in _segment_by_mask(msg.ids, msg.ignore_mask):
        s = tokenizer.decode(ids)
        s = s.replace('\t', '→   ')
        style = STYLE_MASK if is_masked else STYLE_KEEP
        text.append(s, style=style)
    return text


def render_stats(sess: Session) -> Table:
    total = len(sess)
    kept = sum(1 for x in sess.ignore_mask if not x)
    masked = total - kept
    pct = (kept / total * 100) if total else 0.0

    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("total")
    table.add_column("kept (learn)")
    table.add_column("masked (-100)")
    table.add_column("learn ratio")
    table.add_row(
        str(total),
        f"[green]{kept}[/]",
        f"[grey42]{masked}[/]",
        f"[bold]{pct:5.1f}%[/]",
    )
    return table


def show_case(title: str, sess: Session, tokenizer: PackedTokenizer) -> None:
    console.print(Rule(f"[bold magenta]{title}[/]", style="magenta"))

    for i, msg in enumerate(sess.messages):
        header = f"[{i}] role=[bold]{msg.role}[/]  len={len(msg)}"
        console.print(Panel(
            render_message(msg, tokenizer),
            title=header, title_align="left",
            border_style="blue", padding=(0, 1),
        ))

    console.print(render_stats(sess))
    console.print()


# ---------- cases ----------
def case_default_chat(tk):
    sess = Session(tk)
    sess.add_message({'role': 'system', 'content': 'You are a helpful assistant.'})
    sess.add_message({'role': 'user',   'content': '你好，请介绍一下自己。'})
    sess.add_message({
        'role': 'model',
        'thought': '用户在打招呼，应友好回应。',
        'content': '我是一个由林翰打造的语言模型，叫羽田舟。',
    })
    show_case("Case 1 · 默认策略 (system/user=all, model=content)", sess, tk)

def batch(tk):
    sess = Session(tk)
    sess.add_messages([{"role": "system", "content": "", "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"generate_image\", \"description\": \"根据文本描述生成图片\", \"parameters\": {\"type\": \"object\", \"properties\": {\"prompt\": {\"type\": \"string\", \"description\": \"图片描述，详细描述想要生成的图片内容\"}}, \"required\": [\"prompt\"]}}}]"}, {"role": "user", "content": "画一张夕阳下的海港，用油画风格，要温暖的色调和细腻的笔触"}, {"role": "assistant", "content": "我理解你的请求：将夕阳下的海港以油画风格呈现，强调温暖色调和细腻笔触。", "tool_calls": "[{\"function\": {\"name\": \"generate_image\", \"arguments\": \"{\\\"prompt\\\": \\\"sunset harbor oil painting warm colors detailed brushwork\\\"}\"}}]"}, {"role": "tool", "content": "{\"status\": \"success\", \"message\": \"Done! Here is your image\"}"}, {"role": "assistant", "content": "已生成夕阳下的海港油画风格图像，画面呈现温暖色调和细腻笔触，如你所愿！"}])
    show_case("megs", sess, tk)

def case_policy_none(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user',  'content': '强制学习整条消息。'}, mask='none')
    sess.add_message({'role': 'model', 'content': 'hello world'}, mask='none')
    show_case("Case 2 · policy=none (整条参与 loss)", sess, tk)


def case_policy_all(tk):
    sess = Session(tk)
    sess.add_message({'role': 'model', 'content': 'hello world'}, mask='all')
    show_case("Case 3 · policy=all (整条忽略)", sess, tk)


def case_policy_thought(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user',  'content': '2 + 2 = ?'})
    sess.add_message({
        'role': 'model',
        'thought': '计算 2+2，结果为 4。',
        'content': '答案是 4。',
    }, mask='thought')
    show_case("Case 4 · policy=thought (仅 [cot_start]..[cot_end] 之间)", sess, tk)


def case_policy_content(tk):
    sess = Session(tk)
    sess.add_message({
        'role': 'model',
        'thought': '用户在打招呼，应友好回应。',
        'content': 'hi there',
    }, mask='content')
    show_case("Case 5 · policy=content (从 [cot_start] 到句末)", sess, tk)


def case_fim(tk):
    sess = Session(tk)
    sess.add_message({
        'role':   'fim',
        'prefix': 'def add(a, b):\n    ',
        'suffix': '\n    return result',
        'middle': 'result = a + b',
    })
    show_case("Case 6 · FIM (仅 [fim_mid]..[im_end])", sess, tk)


def case_padding(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user',  'content': 'hi'})
    sess.add_message({'role': 'model', 'content': 'hello'})
    sess.pad_to(96)
    show_case("Case 7 · pad_to(96)", sess, tk)


def case_generation_prompt(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user', 'content': '你好'})
    sess.add_generation_prompt(enable_thinking=True)
    show_case("Case 8 · add_generation_prompt(enable_thinking=True)", sess, tk)


def case_explicit_mask(tk):
    """用显式布尔列表做 mask（演示高级用法）。"""
    sess = Session(tk)
    sess.add_message({'role': 'user', 'content': '只学最后几个 token'})
    ids = sess.messages[0].ids
    # 前 70% 忽略，后 30% 学习
    cutoff = int(len(ids) * 0.7)
    explicit = [True] * cutoff + [False] * (len(ids) - cutoff)
    sess.messages[0].ignore_mask = explicit
    show_case("Case 9 · 显式布尔 mask (前 70% 忽略)", sess, tk)

def case_generation_prompt(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user', 'content': '你好'})
    sess.add_generation_prompt(enable_thinking=True)
    show_case("Case 8 · add_generation_prompt(enable_thinking=True)", sess, tk)


def case_gen_prompt_invariant(tk):
    """验证 gen_prompt 始终在逻辑消息末尾（其后仅允许 padding）。"""
    sess = Session(tk)
    sess.add_message({'role': 'user', 'content': 'hi'})
    sess.add_generation_prompt(disable_thinking=True)
    # 此时再追加一条消息，gen_prompt 应自动让位到逻辑末尾
    sess.add_message({'role': 'model', 'content': 'hello'})
    sess.pad_to(64)  # right pad 落在绝对末尾
    show_case("Case 10 · gen_prompt 不变式 (逻辑末尾)", sess, tk)

    # 断言：gen_prompt 之后只允许 padding
    roles = [m.role for m in sess.messages]
    gen_idx = roles.index('__gen_prompt__')
    tail = roles[gen_idx + 1:]
    assert all(r == '__padding__' for r in tail), \
        f'gen_prompt 之后只允许 padding，实际为 {tail}'
    # 且真实消息都在 gen_prompt 之前
    head = roles[:gen_idx]
    assert '__gen_prompt__' not in head, '不能出现多个 gen_prompt'
    assert '__padding__' not in head or tail == [], \
        'right-pad 不应出现在 gen_prompt 之前'

    console.print("[green]✓ 不变式验证通过[/]")

def case_pad_left(tk):
    sess = Session(tk)
    sess.add_message({'role': 'user', 'content': 'hi'})
    sess.add_generation_prompt(enable_thinking=True)
    sess.pad_to(32, side='left')
    show_case("Case 11 · pad_to(32, side='left') 批量推理左填充", sess, tk)

# ---------- main ----------
def legend():
    console.print(Panel(
        Text.assemble(
            ("■ 绿色", STYLE_KEEP), ("  = 参与 loss (label 保留真实 id)\n"),
            ("■ 灰色", STYLE_MASK), ("  = 被 mask (label = -100)"),
        ),
        title="图例", border_style="yellow", padding=(0, 1),
    ))


def main():
    tokenizer = load_tokenizer()
    tokenizer.reset_chat_template()
    legend()
    console.print()

    cases = [
        case_default_chat,
        case_policy_none,
        case_policy_all,
        case_policy_thought,
        case_policy_content,
        case_fim,
        case_padding,
        case_generation_prompt,
        case_explicit_mask,
        case_generation_prompt,
        case_gen_prompt_invariant,
        case_pad_left,
        batch
    ]
    for c in cases:
        c(tokenizer)


if __name__ == '__main__':
    main()