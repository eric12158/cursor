local HMI_IP = "192.168.192.25"
local HMI_PORT = 9000
local HMI_CONNECT_TIMEOUT_SEC = 3
local HMI_READ_TIMEOUT_SEC = 2
local HMI_CONNECTED_FLAG = "ROBOT_CONNECTED"

local M = rawget(_G, "HMI_SOCKET") or {}

local function log(msg)
    textmsg("[int] " .. tostring(msg))
end

function M.ensure_connected()
    if M.client then
        return true
    end

    local ok, socket_or_err = pcall(require, "socket")
    if not ok then
        log("require(\"socket\") failed: " .. tostring(socket_or_err))
        return false
    end

    local client, create_err = socket_or_err.tcp()
    if not client then
        log("socket.tcp() failed: " .. tostring(create_err))
        return false
    end

    client:settimeout(HMI_CONNECT_TIMEOUT_SEC)

    local connected, connect_err = client:connect(HMI_IP, HMI_PORT)
    if connected ~= 1 and connected ~= true then
        log("connect failed: " .. tostring(connect_err))
        client:close()
        return false
    end

    client:settimeout(HMI_READ_TIMEOUT_SEC)
    M.client = client
    log("connected to HMI " .. HMI_IP .. ":" .. tostring(HMI_PORT))
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

    local line, recv_err, partial = M.client:receive("*l")
    if line then
        log("rx <- " .. tostring(line))
        return line
    end

    if recv_err == "timeout" and partial and partial ~= "" then
        log("rx partial <- " .. tostring(partial))
        return partial
    end

    if recv_err ~= "timeout" then
        log("receive failed: " .. tostring(recv_err))
    end

    return nil
end

function M.send_and_wait(payload)
    if not M.ensure_connected() then
        return nil
    end

    if not M.send(payload) then
        return nil
    end

    return M.recv_line()
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
