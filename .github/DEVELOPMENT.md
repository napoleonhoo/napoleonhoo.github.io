# 维护手册（SOP）

本文件放在 `.github/` 下，Jekyll 不会把它渲染成网站页面。

---

## 0. 一次性环境准备

GitHub Pages 的构建机用的是 Ruby 3.3 + `github-pages` gem，本地必须对齐，
否则会出现「本地能构建、线上失败」或反过来的情况。

```bash
# 1) 装 rbenv 并在 shell 中启用（只需做一次）
brew install rbenv ruby-build
echo 'eval "$(rbenv init - zsh)"' >> ~/.zshrc
exec zsh

# 2) 安装 .ruby-version 指定的版本
cd ~/Documents/GitHub/napoleonhoo.github.io
rbenv install "$(cat .ruby-version)"

# 3) 装依赖
gem install bundler
bundle install
```

验证（必须是 3.3.5，不是系统自带的 2.6 或 Homebrew 的 4.x）：

```bash
ruby -v        # => ruby 3.3.5
which ruby     # => ~/.rbenv/shims/ruby
```

> 版本号只在 `.ruby-version` 里维护一处，CI 也读同一个文件。
> 想升级 Ruby 就改这个文件，本地和 CI 一起变。

---

## 1. 写作与本地预览

```bash
bundle exec jekyll serve
# 浏览器打开 http://127.0.0.1:4000
```

常用参数：

| 命令 | 用途 |
| --- | --- |
| `bundle exec jekyll serve --livereload` | 存盘后自动刷新浏览器 |
| `bundle exec jekyll serve --incremental` | 增量构建，文件多时快很多 |
| `bundle exec jekyll build --trace` | 只构建不起服务，报错时打完整堆栈 |

**本地预览有两个盲区，必须靠第 2 节的脚本补**：

1. macOS 文件系统**大小写不敏感**，`blog/Model` 与 `blog/model` 在本地是同一个
   目录，链接点着正常；而线上 Linux 是两个目录，直接 404。
2. 只看首页点得通的几个链接，不会发现深层文章里的坏链接。

---

## 2. 提交前检查（SOP 核心）

每次 `git commit` 前，在仓库根目录依次执行以下三步。
**一行跑完全部**（任一步失败就停下）：

```bash
ruby .github/scripts/check-liquid.rb && \
ruby .github/scripts/check-links.rb && \
bundle exec jekyll build --trace
```

全部通过的输出形如：

```
已检查 267 个文件。
通过：没有发现裸露的 Liquid 分隔符。
已检查 189 条站内相对链接（266 个文件）。
通过：全部命中真实文件（大小写敏感）。
...
                    done in 2.4 seconds.
```

### 本地构建可以忽略的两条 warning

这两条只在本地出现，线上构建机有 API 凭据，不会报，**不用管**：

```
To use retry middleware with Faraday v2.0+, install `faraday-retry` gem
GitHub Metadata: No GitHub API authentication could be found. / 403 - API rate limit exceeded
```

原因是 `jekyll-github-metadata` 插件要调 GitHub API 取仓库信息（owner、
repository_url 等），本地匿名请求会撞限流。它只影响页脚那几个 `site.github.*`
字段，不影响文章内容。

想彻底消掉可以给本地设个只读 token：

```bash
export JEKYLL_GITHUB_TOKEN=ghp_xxx   # 建议写进 ~/.zshrc
```

### 步骤 1：Liquid 分隔符检查

```bash
ruby .github/scripts/check-liquid.rb
```

**查什么**：Markdown 里裸露的 `{{` 和 `{%`。

**为什么**：Jekyll 的渲染顺序是 **Liquid → Markdown**，所以在 Liquid 阶段还不存在
「代码块」这个概念，` ``` ` 围栏里的 `{{` 一样会被当成变量解析。后果有两种：

- `{{` 没有配对的 `}}` → 抛 `Liquid::SyntaxError`，**整个构建失败，站点停更**；
- `{{` 恰好闭合 → 被当变量求值，查不到就渲染成空字符串，**代码被静默吃掉一截**，
  而构建显示"成功"。

写 C++ 聚合初始化（`{{1, "one"}, {2, "two"}}`）、Vue / Go template / Jinja2
模板的文章必踩。

**怎么修**：用 `raw` 标签包住，标签写在 ` ``` ` **外面**（写在里面会在代码块中留下空行）：

<!-- liquid-ok -->
    {% raw %}
    ``` cpp
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}};
    ```
    {% endraw %}

确实是有意使用的 Liquid，在该行加 `<!-- liquid-ok -->` 豁免。

### 步骤 2：站内链接检查

```bash
ruby .github/scripts/check-links.rb
```

**查什么**：所有 `[x](./path)`、`![x](./path)` 形式的站内相对链接是否指向真实文件。

**为什么**：它不读文件系统，而是读 `git ls-files` 的路径集合——**大小写敏感**，
等价于 Linux 构建机上 checkout 出来的结果。这是唯一能在本地发现大小写问题的手段。

常见坏链接成因：

| 成因 | 例子 |
| --- | --- |
| 大小写拼错（最隐蔽） | `blog/model/` vs `blog/Model/` |
| 相对层级多写一层 | 在 `blog/Cpp/` 里写 `../blog/X` → 解析成 `blog/blog/X` |
| 目标改名后引用方没跟着改 | `Ch2.7_PosixEnv` 实际叫 `Ch2.7_EnvPosix` |
| 外链误加 `./` 前缀 | `./https://example.com` |

### 步骤 3：完整构建

```bash
bundle exec jekyll build --trace
```

**查什么**：前两个脚本抓不到的问题——缺失的 layout、坏掉的 front matter、
主题/插件报错等。

除上面那两条 API 噪音外，要求**零 error 且零 warning**。典型 warning：

```
Build Warning: Layout 'post' requested in xxx.md does not exist.
```

本仓库只有 `_layouts/default.html` 一个布局，front matter 一律写
`layout: default`（`post` / `page` / `home` 都不存在）。

---

## 3. 写作约定（照着做可绕开上面大部分坑）

**front matter**：每篇文章开头写

```yaml
---
layout: default
title: 文章标题
---
```

- `layout` 只能是 `default`；
- 注意别把 `title` 拼成 `titile`（历史上错了 9 处，拼错会导致标题取不到）；
- `title` 别从别的文件复制后忘了改（历史上分布式系统的索引标题是 `Go STL`）。

**命名**：

- 目录和文件名**不要用空格**（URL 会变成一长串 `%20`），用下划线，
  例如 `MIT_6.172_Performance_Engineering_of_Software_Systems`；
- 大小写一旦定了就别改。macOS 本地改大小写 git 往往感知不到，
  需要 `git mv old new` 显式操作。

**新建目录**：

1. 建 `0_content.md` 作为该目录的索引页；
2. 在根目录 `index.md` 里加入口，否则是孤岛页面（站内无任何链接可达）。

**代码块含 `{{`**：按第 2 节步骤 1 的方式用 `raw` 包起来。

---

## 4. 推送与部署确认

```bash
git add -A
git commit -m "..."
git push
```

推送后：

1. **CI**：`.github/workflows/ci.yml` 会跑 `liquid` 与 `build` 两个 job
   （`https://github.com/napoleonhoo/napoleonhoo.github.io/actions`）；
2. **部署**：站点由 GitHub 内置的 `pages-build-deployment` 流程发布，
   和上面的 CI 是**两条独立流水线**——CI 绿不代表部署成功，要单独确认那条
   `pages build and deployment` 是否通过。

部署失败时，日志里**真正的报错在 `build` job 的日志末尾**，
注解面板上那一大段 debug 日志会被网页截断，容易误判。
例如 `The github-pages gem can't satisfy your Gemfile's dependencies` 只是
**warning**，不是失败原因。

---

## 5. 故障速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 部署失败，日志有 `Liquid syntax error ... was not properly terminated` | 代码块里有未闭合的 `{{` | 步骤 1 |
| 页面上代码缺了一截，构建却成功 | 闭合的 `{{}}` 被当变量渲染成空串 | 步骤 1 |
| 本地点链接正常，线上 404 | 路径大小写不一致 | 步骤 2 |
| `Build Warning: Layout 'xxx' does not exist` | front matter 写了不存在的 layout | 改成 `layout: default` |
| URL 里一堆 `%20` | 文件/目录名带空格 | `git mv` 改成下划线 |
| 本地 `bundle install` 报 Ruby 版本不满足 | 用了系统 2.6 或 Homebrew 4.x | 第 0 节，确认 `which ruby` 指向 rbenv shims |
| CI 的 `jekyll build` 失败，日志有 `No repo name found` | `jekyll-github-metadata` 推断不出仓库名，在 production 下是致命错误 | 已在 `_config.yml` 设 `repository:`，勿删 |
| CI 报 `Unexpected input(s) 'xxx'` | 给 action 传了它不支持的参数（会被忽略，可能引发连带失败） | 以 action 的 `action.yml` 为准，别凭印象写参数 |
| 页脚日期比预期早一天 | 构建机是 UTC | 已在 `_config.yml` 设 `timezone: Asia/Shanghai` |

---

## 附：仓库关键文件

| 路径 | 作用 |
| --- | --- |
| `_config.yml` | 站点配置（主题、时区） |
| `_layouts/default.html` | 全站唯一布局，含页脚 |
| `_sass/` | 主题样式覆盖 |
| `index.md` | 首页与各栏目入口 |
| `.ruby-version` | Ruby 版本，本地与 CI 共用 |
| `Gemfile` / `Gemfile.lock` | 依赖，对齐线上 `github-pages` |
| `.github/scripts/check-liquid.rb` | 提交前检查：Liquid 分隔符 |
| `.github/scripts/check-links.rb` | 提交前检查：站内链接 |
| `.github/workflows/ci.yml` | CI：lint + 构建校验 |
