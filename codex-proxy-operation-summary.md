# Codex App + WSL2 代理配置操作总结

## 1. 环境确认

- 当前 WSL 用户目录：`/home/blybq`
- 当前项目目录：`/home/blybq/code-project/cs336`
- WSL 发行版：Ubuntu
- 当前 Codex App 使用的是 WSL bundled CLI，而不是普通 PATH 中的 Linux CLI。
- 发现的 Codex 路径：
  - 用户安装版：`/home/blybq/.local/bin/codex`
  - Codex App bundled WSL CLI：`/mnt/c/Users/12391/.codex/bin/wsl/<版本目录>/codex`

## 2. 代理方案

目标是保持：

```ini
[wsl2]
autoProxy=false
networkingMode=mirrored
```

并只让 Codex WSL app-server 使用 Clash 代理，不让整个 WSL 环境自动继承代理。

默认代理地址设置为：

```text
http://127.0.0.1:7897
```

这里假设 Clash 的 7897 是 HTTP 或 mixed 代理端口。

## 3. 创建的 wrapper

在项目目录创建了：

```text
/home/blybq/code-project/cs336/codex-wsl-proxy
```

该 wrapper 会：

1. 设置 `HTTP_PROXY`、`HTTPS_PROXY` 及小写版本；
2. 设置 `NO_PROXY` 为 localhost 地址；
3. 清除可能冲突的 `ALL_PROXY`；
4. 自动寻找当前 Codex App bundled WSL CLI；
5. 使用 `exec` 将参数原样传给真实 Codex CLI；
6. 支持 `CODEX_WSL_PROXY_URL` 覆盖默认代理地址；
7. 支持 `CODEX_WSL_PROXY_DRY_RUN=1` 进行路径检查。

wrapper 已通过 Bash 语法检查，并成功执行版本测试。

## 4. 复制到 Windows 用户目录

wrapper 已复制到：

```text
/mnt/c/Users/12391/codex-wsl-proxy
C:\Users\12391\codex-wsl-proxy
```

## 5. 设置的 Windows 用户环境变量

已设置：

```text
CODEX_CLI_PATH=C:\Users\12391\codex-wsl-proxy
```

并通过以下注册表查询确认：

```text
HKCU\Environment\CODEX_CLI_PATH
```

## 6. 验证情况

- wrapper 文件存在且可执行；
- wrapper 的 dry-run 结果确认默认代理为 `http://127.0.0.1:7897`；
- wrapper 能找到当前 bundled WSL CLI；
- wrapper 直接执行 `--version` 成功；
- Codex App 重启后已经能够正常产生模型 token，说明 WSL Codex 请求链路已经恢复。

严格确认是否经过 Clash，仍应查看 Clash 连接日志，或在 WSL 中检查运行中 app-server 的代理环境变量。

## 7. 手动测试命令

确认 Windows 用户变量：

```powershell
reg query HKCU\Environment /v CODEX_CLI_PATH
```

确认 WSL mirrored 模式可以访问 Clash HTTP 端口：

```bash
curl -v -x http://127.0.0.1:7897 -I https://chatgpt.com
```

确认 app-server 代理环境：

```bash
pgrep -af 'codex.*app-server'
pid=$(pgrep -n -f 'codex.*app-server')
tr '\0' '\n' < "/proc/$pid/environ" | grep -iE '^(HTTP|HTTPS|http|https|NO_PROXY|no_proxy)='
```

如果 7897 是 SOCKS-only 端口，则测试命令应改为：

```bash
curl -v --socks5-hostname 127.0.0.1:7897 -I https://chatgpt.com
```

## 8. 回滚方式

如果 Codex App 更新后不能启动，可以将 `CODEX_CLI_PATH` 恢复为原 bundled Windows CLI 路径，或删除该用户环境变量后重启 Codex App。

注意：`CODEX_CLI_PATH` 是启动器兼容机制，不是当前 Codex 官方稳定公开配置，未来 App 更新后可能需要重新设置。
