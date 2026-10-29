#!/usr/bin/env ruby
# frozen_string_literal: true

# 校验 Markdown 里的站内相对链接是否指向真实存在的文件。
#
# 为什么不能只靠本地预览：
#   macOS 的文件系统默认大小写不敏感，blog/Model 和 blog/model 在本地是同一个
#   目录，浏览器点链接一切正常；但 GitHub 的 Linux 构建机大小写敏感，是两个
#   独立目录，于是线上 404。这类 bug 本地 100% 看不出来。
#
# 所以这里不读文件系统，而是用 git 的路径集合（大小写敏感，等价于 Linux
# runner 上 checkout 出来的结果）来判断链接目标是否存在。
#
# 用法：
#   ruby .github/scripts/check-links.rb
#
# 退出码：0 = 通过，1 = 发现坏链接。
#
# 说明：
#   - 只检查以 ./ 或 ../ 开头的站内相对链接，外部 http(s) 链接不做可达性探测；
#   - 链接里的 #anchor 和 ?query 会被忽略；
#   - 覆盖普通链接 [x](...) 与图片 ![x](...)。

require 'set'

# 定位仓库根目录，这样在任意子目录下都能执行
repo = `git rev-parse --show-toplevel`.strip
if repo.empty?
  warn '错误：当前不在 git 仓库内。'
  exit 1
end
Dir.chdir(repo)

# 已跟踪 + 待新增（untracked），再排除工作区已删除的，
# 反映“提交之后”的最终状态，避免误报刚新建还没 git add 的文件。
tracked = Set.new(
  (`git ls-files`.split("\n") +
   `git ls-files --others --exclude-standard`.split("\n"))
  .select { |f| File.exist?(f) }
)

# Jekyll 会忽略点开头的目录（.github 等），这些不是网站内容，不参与检查
def site_content?(path)
  !path.split(File::SEPARATOR).any? { |seg| seg.start_with?('.') }
end

# 链接写的是 .html（渲染后的产物），要回推到它对应的源文件
def candidates(target)
  base = target.sub(/\.html\z/, '')
  ["#{base}.md", "#{base}.markdown", "#{base}.html", target]
end

bad = []
checked = 0

tracked.select { |f| f =~ /\.(md|markdown)\z/ && site_content?(f) }.sort.each do |src|
  text = File.read(src, encoding: 'UTF-8').scrub('?')
  dir = File.dirname(src)

  # [text](./path) 与 ![alt](./path)，只取 ./ 或 ../ 开头的
  text.scan(/\[[^\]]*\]\((\.{1,2}\/[^)\s]+)\)/).flatten.each do |raw|
    link = raw.sub(/[#?].*\z/, '')          # 去掉 #anchor / ?query
    next if link.empty?

    checked += 1
    resolved = File.expand_path(link, File.join(repo, dir)).sub("#{repo}/", '')
    next if candidates(resolved).any? { |c| tracked.include?(c) }

    bad << [src, raw, resolved]
  end
end

puts "已检查 #{checked} 条站内相对链接（#{tracked.count { |f| f =~ /\.(md|markdown)\z/ && site_content?(f) }} 个文件）。"

if bad.empty?
  puts '通过：全部命中真实文件（大小写敏感）。'
  exit 0
end

puts "发现 #{bad.size} 条坏链接："
bad.each do |src, link, resolved|
  puts "  #{src}"
  puts "      链接: #{link}"
  puts "      解析: #{resolved}  <- git 中不存在"
  puts "::error file=#{src}::坏链接 #{link} -> #{resolved}" if ENV['GITHUB_ACTIONS']
end

warn <<~MSG

  排查提示：
    - 大小写拼错（macOS 本地不敏感，线上敏感，最容易漏）
    - 相对路径层级写错，例如在 blog/Cpp/ 里写 ../blog/X 实际会解析到 blog/blog/X
    - 目标文件改名或移动后，引用方没跟着改
    - 外部链接误加了 ./ 前缀
MSG

exit 1
