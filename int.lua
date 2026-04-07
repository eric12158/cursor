local host = "192.168.192.25"
local port = 9000
local HMI_CONNECTED_FLAG = "client successfully connect!"

local M = rawget(_G, "HMI_SOCKET") or {}

local function log(msg)
    textmsg("[int] " .. tostring(msg))
end

function M.ensure_connected()
    if M.client then
        return true
    end

    local ok, socket = pcall(require, "socket")
    if not ok then
        log("require(\"socket\") failed: " .. tostring(socket))
        return false
    end

    local client, err = socket.connect(host, port)
    if not client then
        log("connect failed: " .. tostring(err))
        return false
    end

    client:settimeout(0)
    M.client = client
    log("client connected to HMI " .. host .. ":" .. tostring(port))
    return true
end

function M.send(payload)
    if not M.client then
        log("send skipped: socket is not connected")
        return false
    end

    local ok, send_err, bytes_sent = M.client:send(tostring(payload) .. "\n")
    if not ok then
        log("send failed: " .. tostring(send_err) .. ", bytes_sent=" .. tostring(bytes_sent))
        return false
    end

    return true
end

function M.recv_line()
    if not M.client then
        log("receive skipped: socket is not connected")
        return nil
    end

    local line, recv_err, partial = M.client:receive()
    if line then
        log("rx <- " .. tostring(line))
        return line
    end

    if partial and partial ~= "" then
        log("rx partial <- " .. tostring(partial))
        return partial
    end

    if recv_err and recv_err ~= "timeout" then
        log("receive failed: " .. tostring(recv_err))
    end

    return nil
end

function M.close()
    if M.client then
        pcall(function()
            M.client:close()
        end)
    end

    M.client = nil
end

_G.HMI_SOCKET = M

if M.ensure_connected() then
    M.send(HMI_CONNECTED_FLAG)
end
