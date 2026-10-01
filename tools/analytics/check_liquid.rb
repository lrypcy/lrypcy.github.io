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

if failures.empty?
  puts "\n全部通过。"
  exit 0
else
  puts "\n失败项："
  failures.each { |f| puts "  - #{f}" }
  exit 1
end
