#!/usr/bin/env ruby
# frozen_string_literal: true

# 离线校验 _includes/analytics.html 的 Liquid 语法与渲染结果。
#
# 为什么需要它：本站由 GitHub Pages 构建，本地 Jekyll 不可用（缺 bundler 2.5.4），
# 一旦 include 里写错一个标签，整站构建会直接失败。这个脚本用 liquid gem 做
# 「parse 语法检查 + 多配置渲染」，把风险挡在 push 之前。
#
# 运行：ruby tools/analytics/check_liquid.rb

require 'liquid'
require 'json'

# Jekyll 的 jsonify 等价于 JSON.generate（对字符串会带上双引号）。
module JekyllishFilters
  def jsonify(input)
    JSON.generate(input.to_s)
  end
end
Liquid::Template.register_filter(JekyllishFilters)

INCLUDE_PATH = File.expand_path('../../_includes/analytics.html', __dir__)

CONFIGS = {
  '未部署：enabled=false, endpoint 为空' => {
    'analytics' => { 'enabled' => false, 'endpoint' => '' }
  },
  '未部署：enabled=true 但 endpoint 为空' => {
    'analytics' => { 'enabled' => true, 'endpoint' => '' }
  },
  '已部署：字段完整' => {
    'analytics' => {
      'enabled' => true,
      'endpoint' => 'https://pcy-analytics.example.workers.dev/',
      'production_host' => 'lrypcy.github.io',
      'footer_label' => '总访问',
      'footer_unit' => '次',
      'respect_dnt' => false
    }
  },
  '已部署：只填了必填项' => {
    'analytics' => {
      'enabled' => true,
      'endpoint' => 'https://pcy-analytics.example.workers.dev'
    }
  },
  '配置缺失：site.analytics 不存在' => {}
}.freeze

source = File.read(INCLUDE_PATH, encoding: 'UTF-8')

begin
  template = Liquid::Template.parse(source)
rescue Liquid::SyntaxError => e
  warn "Liquid 语法错误：#{e.message}"
  exit 1
end
puts "Liquid parse: OK（#{source.lines.size} 行）"

failures = []

CONFIGS.each do |name, config|
  # 模板里读的是 site.analytics，CONFIGS 的每一项恰好就是 site 的内容
  rendered = template.render('site' => config)
  stripped = rendered.strip

  if name.start_with?('未部署', '配置缺失')
    if stripped.empty?
      puts "PASS  #{name} → 不输出任何内容"
    else
      failures << "#{name}：预期无输出，实际输出了 #{stripped.bytesize} 字节"
    end
    next
  end

  checks = {
    '含采集端点 /c' => stripped.include?("'/c'") || stripped.include?('/c'),
    '含计数端点 /total' => stripped.include?('/total'),
    '含页脚元素 id' => stripped.include?('pcy-visit-total'),
    '含 sendBeacon 分支' => stripped.include?('sendBeacon'),
    '含 fetch 兜底' => stripped.include?('keepalive'),
    '含生产域名判定' => stripped.include?('lrypcy.github.io'),
    '含 notrack 逃生口' => stripped.include?('notrack'),
    '同时只有一段 script' => stripped.scan('<script').size == 1
  }

  bad = checks.reject { |_, ok| ok }.keys
  if bad.empty?
    puts "PASS  #{name} → 输出 #{stripped.bytesize} 字节，#{checks.size} 项检查通过"
  else
    failures << "#{name}：未通过 #{bad.join('、')}"
  end
end

# endpoint 末尾斜杠应被去掉，否则会拼出 //c
rendered = Liquid::Template.parse(source).render('site' => CONFIGS['已部署：字段完整'])
failures << 'endpoint 尾部斜杠未被处理' if rendered.include?("workers.dev/'") || rendered.include?('workers.dev//')

# ---------------------------------------------------------------------------
# 站点其它含 Liquid 的源文件也要做语法检查。
# 本地 Jekyll 跑不起来，一个没闭合的 {% if %} 会直接把整站构建打挂，
# 而这类错误在浏览器里只表现为「站点没更新」，很难定位。
# ---------------------------------------------------------------------------
ROOT = File.expand_path('../..', __dir__)

liquid_sources = Dir[
  File.join(ROOT, '*.md'),
  File.join(ROOT, '*.html'),
  File.join(ROOT, '_layouts', '*.html'),
  File.join(ROOT, '_includes', '*.html')
].sort

unclosed = []
liquid_sources.each do |path|
  text = File.read(path, encoding: 'UTF-8')
  next unless text.include?('{%') || text.include?('{{')

  begin
    Liquid::Template.parse(text)
  rescue Liquid::SyntaxError => e
    unclosed << "#{path.delete_prefix("#{ROOT}/")}：#{e.message}"
  end
end

if unclosed.empty?
  puts "\nLiquid 语法检查：#{liquid_sources.size} 个站点源文件全部通过"
else
  failures.concat(unclosed.map { |u| "Liquid 语法错误 → #{u}" })
end

# ---------------------------------------------------------------------------
# 关于页内嵌看板：单独渲染那一段，确认开关真的生效、endpoint 真的被替换进去。
# 整页渲染需要 Jekyll 的 date / plus / divided_by 等过滤器，所以只截取这一段。
# ---------------------------------------------------------------------------
about = File.read(File.join(ROOT, 'about.md'), encoding: 'UTF-8')
block = about[/\{%\s*assign ana = site\.analytics\s*%\}.*?\{%\s*endif\s*%\}/m]
if block.nil?
  failures << 'about.md：找不到访问统计区块（改过标记就用这个脚本重新对齐）'
else
  # 用与 _config.yml 同形的 endpoint（不带尾斜杠）。带尾斜杠的写法由 Worker 侧的
  # 路径归一化兜底，另有单测覆盖，这里不重复。
  about_config = Marshal.load(Marshal.dump(CONFIGS['已部署：字段完整']))
  about_config['analytics']['endpoint'] = about_config['analytics']['endpoint'].sub(%r{/+\z}, '')

  enabled = Liquid::Template.parse(block).render('site' => about_config)
  disabled = Liquid::Template
             .parse(block)
             .render('site' => CONFIGS['配置缺失：site.analytics 不存在'])

  about_checks = {
    '开启了才输出 iframe' => enabled.include?('<iframe id="pcy-stats-frame"'),
    'iframe 指向 /stats?embed=1' => enabled.include?('workers.dev/stats?embed=1'),
    'endpoint 已被替换（不留 Liquid 痕迹）' => !enabled.include?('{{') && !enabled.include?('{%'),
    '带高度回填脚本' => enabled.include?('pcy-analytics:height'),
    '不暴露口令' => !enabled.include?('token='),
    '关掉统计就不输出' => disabled.strip.empty?
  }
  bad = about_checks.reject { |_, ok| ok }.keys
  if bad.empty?
    puts "PASS  about.md 访问统计区块 → #{about_checks.size} 项检查通过"
  else
    failures << "about.md 访问统计区块：未通过 #{bad.join('、')}"
  end
end

if failures.empty?
  puts "\n全部通过。"
  exit 0
else
  puts "\n失败项："
  failures.each { |f| puts "  - #{f}" }
  exit 1
end
