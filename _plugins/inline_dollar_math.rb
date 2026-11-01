# 行内 $...$ → kramdown 认识的等价写法
#
# 目的：让 _posts 里的行内公式直接用 LaTeX / 知乎的通行写法 $...$，同一份 Markdown
# 既能被本站渲染，也能原样贴到知乎。
#
# 为什么不能直接写：kramdown 只把 `$$...$$` 认作数学元素，并把里面的内容原样保护；
# 单 `$...$` 对它只是普通文本，会先完整跑一遍 Markdown。实测（kramdown 2.4 + GFM）：
#
#   $c$ 裸写                  → 原样输出 $c$，内容已被 Markdown 处理过
#   $\left\{ a \right\}$      → $\left{ a \right}$      \{ 的反斜杠被吃掉
#   $\begin{aligned}a\\b…$    → $…a\b…$                 \\ 被压成单个 \
#   $V^*$ 与 $\pi^*$          → $V^<em>$ 与 $\pi^</em>$  两个 * 配成斜体
#   $p(x|y)$                  → 变成表格
#   $a<b>c$                   → $a<b>c$ + 多余的 </b>   HTML 标签被解析
#   $f'(x)$                   → $f’(x)$                 smart_quotes 把撇号换成 U+2019
#
# 最后这条最阴：kramdown 的 smart_quotes 默认开着，会把公式里的 `'` 变成 ‘ ’
# （`\alpha'_t` 显示成 $\alpha’_t$，MathJax 认不出来）。写在 `$$...$$` 里则不会被碰。
#
# 也就是说：不是「$ 不被识别」这么简单，而是公式内容被 Markdown 拆碎了，之后 MathJax
# 拿到的已经不是原来的 TeX。任何 kramdown 不认识的新定界符都会有同样的问题，唯一
# 的办法是在 Markdown 之前把内容保护起来 —— 就是这个钩子干的事。
#
# 做法：把行内 `$...$` 改写成同样夹在句中的 `$$...$$`。kramdown 把它渲染成行内
# `\(...\)`，与 $...$ 的渲染结果完全一致；行间 `$$`、代码块、行内代码、`\$` 全部原样保留。
#
# 生效前提：GitHub Pages 必须以 GitHub Actions 方式构建。legacy 分支构建跑在 safe mode，
# 会静默忽略 _plugins/（站点照常出，只是这个钩子不起作用，退化成裸 $ 行为）。
# 设置位置：仓库 Settings → Pages → Source 选 GitHub Actions，见 .github/workflows/pages.yml。

module Jekyll
  module InlineDollarMath
    # 私有区字符，正文里不会出现，避免与内容冲突
    SENTINEL = "\u{E000}".freeze
    PLACEHOLDER = /#{SENTINEL}(\d+)#{SENTINEL}/.freeze

    # 行内 $...$
    #   (?<![\\$\w])  左边界不是反斜杠 / 另一个 $ / 字母数字（避开 $$ 与 $var）
    #   (?!\s)        开 $ 后面不能紧跟空白
    #   ([^$\n]+?)    内容不跨行、不含 $
    #   (?<!\s)       关 $ 前面不能是空白
    #   (?![\d\w])    关 $ 后面不能紧跟数字字母
    #
    # 金额靠「关 $ 后面不能紧跟数字」挡住：`成本 $5，收益 $10` 里第二个 $ 后面是数字，
    # 不构成闭合，整段不转换。正文真要写美元符号时用 \$ 显式转义。
    INLINE = %r{(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\d\w])}.freeze

    def self.convert(text)
      return text if text.nil? || !text.include?("$")

      store = []
      stash = lambda do |s|
        store << s
        "#{SENTINEL}#{store.size - 1}#{SENTINEL}"
      end

      t = text.dup
      # 1. 围栏代码块、行内代码 —— 里面的 $ 一律不动
      t = t.gsub(/^```.*?^```/m) { |m| stash.call(m) }
      t = t.gsub(/^~~~.*?^~~~/m) { |m| stash.call(m) }
      t = t.gsub(/`[^`\n]+`/) { |m| stash.call(m) }
      # 2. 行间 $$...$$（单行、三行式都在内）
      t = t.gsub(/\$\$[\s\S]*?\$\$/) { |m| stash.call(m) }
      # 3. 转义的 \$
      t = t.gsub(/\\\$/) { |m| stash.call(m) }
      # 4. 剩余的单 $ 配对成行内公式
      t = t.gsub(INLINE) { "$$#{Regexp.last_match(1)}$$" }
      # 5. 还原
      t.gsub(PLACEHOLDER) { store[Regexp.last_match(1).to_i] }
    end
  end
end

# posts / pages / 自定义 collection（_daily）都过一遍
%i(posts pages daily).each do |owner|
  Jekyll::Hooks.register owner, :pre_render do |doc|
    doc.content = Jekyll::InlineDollarMath.convert(doc.content)
  end
end
