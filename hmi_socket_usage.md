# HMI Socket 使用说明

适用场景：

- HMI 作为普通 TCP Socket 服务端，监听 `9000`
- `int.lua` 负责初始化连接，并注册全局 `LOG()`
- 业务脚本负责发送业务报文并等待 HMI 回包

## 1. 关键结论

如果初始化脚本里这样写：

- `local client = assert(socket.connect(HmiIp, HmiPort))`

那么这个 `client` 只是当前脚本里的局部变量。

业务脚本里**不能直接访问**这个局部变量，所以要在初始化脚本里把它挂到全局对象上，例如：

- `_G.HMI_SOCKET = M`

这样后面的业务脚本才能通过：

- `local hmi = rawget(_G, "HMI_SOCKET")`

拿到同一个连接对象。

## 2. 初始化脚本应该做什么

初始化脚本只做 3 件事：

1. 连接 HMI
2. 把连接对象保存到 `_G.HMI_SOCKET`
3. 注册全局 `LOG()` 并发送连接成功标志

参考 `int.lua`：

- 使用 `require("socket")`
- 使用 `socket.connect(host, port)`
- 连接成功后 `client:settimeout(0)`
- 注册 `_G.LOG = function(...) ... end`
- 保存到 `_G.HMI_SOCKET`
- 发送 `client successfully connect!`

## 3. 业务脚本应该做什么

业务脚本只做 5 件事：

1. 读取 `_G.HMI_SOCKET`
2. 检查连接对象是否存在
3. 发送业务报文
4. 等待 HMI 回包
5. 超时报错并停止程序

参考 `socketsend.lua`。

## 4. 最小业务代码模板

```lua
local function log(msg)
    textmsg("[socket-send] " .. tostring(msg))
end

local function fail(msg)
    log(msg)
    error(msg)
end

local hmi = rawget(_G, "HMI_SOCKET")
if not hmi then
    fail("HMI socket object not found")
end

local WAIT_REPLY_TIMEOUT_SEC = 120
local tx = "GET_POS"

local ok = hmi.ensure_connected()
if not ok then
    fail("connect failed")
end

if not hmi.send(tx) then
    fail("send failed")
end

log("tx sent, waiting up to " .. tostring(WAIT_REPLY_TIMEOUT_SEC) .. "s for HMI reply")
hmi.client:settimeout(WAIT_REPLY_TIMEOUT_SEC)
local rx, recv_err, partial = hmi.client:receive()
hmi.client:settimeout(0)

if rx then
    log("rx=" .. tostring(rx))
elseif recv_err == "timeout" and partial and partial ~= "" then
    log("rx partial=" .. tostring(partial))
else
    fail("no reply within " .. tostring(WAIT_REPLY_TIMEOUT_SEC) .. "s")
end
```

## 5. 现在用什么指令发送

当前没有固定协议，发送内容由你自己定义。

例如可以发：

- `GET_POS`
- `PICK`
- `DROP`
- `REQ:COORDS`

只要 HMI 端和机器人端约定一致即可。

## 6. HMI 应该怎么回

当前代码用的是：

- `client:send(payload .. "\\n")`
- `client:receive()`

所以建议 HMI 一次回复一条完整文本，例如：

- `OK`
- `100,200,300,180,0,90`
- `PICK_DONE`

## 7. 超时行为

当前业务脚本里：

- 最长等待 120 秒
- 超时后调用 `error(...)`

因此程序会在业务节点报错并停住，不会继续往后执行。

## 8. 当前文件职责

- `int.lua`：初始化连接、注册全局 `LOG()`、保存 `_G.HMI_SOCKET`、发送连接成功标志
- `socketsend.lua`：业务发送、等待回包、打印日志、超时报错
