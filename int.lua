-- HMI socket init / shared helper
-- Port is fixed to the normal TCP socket service on 9000.
-- This script stores the live connection object in _G.HMI_SOCKET.
-- Other ScriptNode files should read _G.HMI_SOCKET directly instead of require("int").

local M = rawget(_G, "HMI_SOCKET") or {}

local socket_lib = M._socket_lib
local client = M._client

local DEFAULT_CFG = {
    ip = "192.168.192.25",
    port = 9000,
    connect_timeout = 3,
    read_timeout = 2,
    connected_flag = "ROBOT_CONNECTED",
}

local function log(msg)
    textmsg("[int] " .. tostring(msg))
end

local function load_socket()
    if socket_lib then
        return socket_lib
    end

    local ok, lib_or_err = pcall(require, "socket")
    if not ok then
        log("require(\"socket\") failed: " .. tostring(lib_or_err))
        return nil
    end

    socket_lib = lib_or_err
    M._socket_lib = socket_lib
    return socket_lib
end

local function merge_cfg(cfg)
    cfg = cfg or {}
    return {
        ip = cfg.ip or DEFAULT_CFG.ip,
        port = cfg.port or DEFAULT_CFG.port,
        connect_timeout = cfg.connect_timeout or DEFAULT_CFG.connect_timeout,
        read_timeout = cfg.read_timeout or DEFAULT_CFG.read_timeout,
        connected_flag = cfg.connected_flag or DEFAULT_CFG.connected_flag,
    }
end

function M.open(cfg)
    if client then
        return true
    end

    local socket = load_socket()
    if not socket then
        return false
    end

    local merged = merge_cfg(cfg)
    local tcp_client, create_err = socket.tcp()
    if not tcp_client then
        log("socket.tcp() failed: " .. tostring(create_err))
        return false
    end

    tcp_client:settimeout(merged.connect_timeout)

    local ok, connect_err = tcp_client:connect(merged.ip, merged.port)
    if ok ~= 1 and ok ~= true then
        log("connect failed: " .. tostring(connect_err))
        tcp_client:close()
        return false
    end

    tcp_client:settimeout(merged.read_timeout)
    client = tcp_client
    M._client = client
    M.cfg = merged

    log("connected to HMI " .. merged.ip .. ":" .. tostring(merged.port))

    if merged.connected_flag then
        M.send(merged.connected_flag)
    end

    return true
end

function M.is_connected()
    return client ~= nil
end

function M.send(payload)
    if not client then
        log("send skipped: socket is not connected")
        return false
    end

    local ok, send_err, bytes_sent = client:send(tostring(payload) .. "\n")
    if not ok then
        log("send failed: " .. tostring(send_err) .. ", bytes_sent=" .. tostring(bytes_sent))
        return false
    end

    log("tx -> " .. tostring(payload))
    return true
end

function M.recv_line()
    if not client then
        log("receive skipped: socket is not connected")
        return nil
    end

    local line, recv_err, partial = client:receive("*l")
    if line then
        M.last_rx = line
        log("rx <- " .. tostring(line))
        return line
    end

    if recv_err == "timeout" and partial and partial ~= "" then
        M.last_rx = partial
        log("rx partial <- " .. tostring(partial))
        return partial
    end

    if recv_err ~= "timeout" then
        log("receive failed: " .. tostring(recv_err))
    end

    return nil
end

function M.request(payload)
    if not M.send(payload) then
        return nil
    end

    return M.recv_line()
end

function M.ensure_connected(cfg)
    if client then
        return true
    end

    return M.open(cfg or DEFAULT_CFG)
end

function M.send_and_wait(payload)
    if not M.ensure_connected() then
        return nil
    end

    return M.request(payload)
end

function M.close()
    if client then
        local ok, close_err = pcall(function()
            client:close()
        end)
        if not ok then
            log("close failed: " .. tostring(close_err))
        end
    end

    client = nil
    M._client = nil
    log("socket closed")
end

_G.HMI_SOCKET = M

-- Run as the initialization script: connect and send the connected flag once.
M.open(DEFAULT_CFG)

return M
