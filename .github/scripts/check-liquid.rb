#!/usr/bin/env ruby
# frozen_string_literal: true

# 检查 Markdown 里裸露的 Liquid 分隔符。
#
# 背景：Jekyll 的渲染顺序是 Liquid -> Markdown，所以 Liquid 阶段还不存在
# “代码块”的概念，``` 围栏里的 {{ 一样会被当成变量解析。后果有两种：
#
#   1. {{ 没有配对的 }}  -> Liquid::SyntaxError，整个 jekyll build 直接失败；
#   2. {{ 恰好闭合       -> 被当变量求值，查不到就渲染成空字符串，
#                           代码被静默吃掉一截，构建却是“成功”的。
#
# 正确写法是用 {% raw %} ... {% endraw %} 把代码块包起来（标签放在 ``` 外面，
# 避免标签被移除后在代码块里留下空行）。
#
# 用法：
#   ruby .github/scripts/check-liquid.rb            # 检查整个仓库
#   ruby .github/scripts/check-liquid.rb a.md b.md  # 只检查指定文件
#
# 退出码：0 = 通过，1 = 发现问题。
#
# 如果某处确实是有意使用的 Liquid，在该行加上 <!-- liquid-ok --> 即可豁免。

require 'strscan'

# Liquid / Jekyll 的合法标签，出现在 raw 块外面是正常的。
ALLOWED_TAGS = %w[
  raw endraw
  highlight endhighlight
  if elsif else endif unless endunless
  case when endcase
  for endfor break continue cycle tablerow endtablerow
  assign capture endcapture increment decrement
  comment endcomment
  include include_relative link post_url
  seo feed_meta gist render form avatar
].freeze

EXEMPT_MARKER = 'liquid-ok'

SKIP_DIRS = %w[_site vendor node_modules .jekyll-cache].freeze

def target_files(argv)
  return argv.select { |f| File.file?(f) } unless argv.empty?

  Dir.glob('**/*.{md,markdown}', File::FNM_DOTMATCH).reject do |path|
    parts = path.split(File::SEPARATOR)
    # 跳过构建产物，以及点开头的目录（.git/.github 等）——Jekyll 不渲染它们，
    # 里面的 {{ 不会进入网站，检查没有意义。与 check-links.rb 的口径保持一致。
    SKIP_DIRS.any? { |d| parts.include?(d) } || parts.any? { |s| s.start_with?('.') }
  end.sort
end

# 扫描单个文件，返回 offense 数组。
def scan_file(path)
  offenses = []
  in_raw = false
  raw_opened_at = nil

  # 显式指定 UTF-8，否则在 locale 为 POSIX/C 的环境（含部分 CI runner）下
  # 读到中文会抛 invalid byte sequence in US-ASCII。scrub 兜住个别坏字节。
  lines = File.read(path, encoding: 'UTF-8').scrub('?').lines

  lines.each_with_index do |line, idx|
    lineno = idx + 1
    exempt = line.include?(EXEMPT_MARKER)
    scanner = StringScanner.new(line)

    until scanner.eos?
      if scanner.scan(/\{%-?\s*raw\s*-?%\}/)
        in_raw = true
        raw_opened_at = lineno
      elsif scanner.scan(/\{%-?\s*endraw\s*-?%\}/)
        in_raw = false
        raw_opened_at = nil
      elsif (tag = scanner.scan(/\{%-?\s*([a-zA-Z_]\w*)/))
        name = tag[/[a-zA-Z_]\w*\z/]
        unless in_raw || exempt || ALLOWED_TAGS.include?(name)
          offenses << [lineno, "未知 Liquid 标签 {% #{name} %}", line]
        end
      elsif scanner.scan(/\{\{/)
        unless in_raw || exempt
          offenses << [lineno, '裸露的 {{（会被 Liquid 当成变量解析）', line]
        end
      else
        scanner.getch
      end
    end
  end

  offenses << [raw_opened_at, '{% raw %} 没有对应的 {% endraw %}', ''] if in_raw
  offenses
end

files = target_files(ARGV)
total = 0

files.each do |path|
  scan_file(path).each do |lineno, reason, line|
    total += 1
    puts "#{path}:#{lineno}: #{reason}"
    puts "    #{line.strip}" unless line.strip.empty?
    # GitHub Actions 的行内标注
    if ENV['GITHUB_ACTIONS']
      puts "::error file=#{path},line=#{lineno}::#{reason}"
    end
  end
end

puts "已检查 #{files.size} 个文件。"

if total.zero?
  puts '通过：没有发现裸露的 Liquid 分隔符。'
  exit 0
end

warn <<~MSG

  发现 #{total} 处问题。修复方法：把受影响的代码块用 raw 标签包起来（标签写在 ``` 外面）：

      {% raw %}
      ``` cpp
      std::map<int, std::string> m = {{1, "one"}, {2, "two"}};
      ```
      {% endraw %}

  若某处确实是有意使用的 Liquid，在该行加 <!-- #{EXEMPT_MARKER} --> 豁免。
MSG

exit 1
