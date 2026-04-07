-- AUBO ARCS/AuboStudio ScriptNode example
-- Purpose:
-- 1) Actively connect to an HMI TCP socket server.
-- 2) Send one line to the HMI and wait for one reply line.
-- 3) Optionally keep the socket open for later use.
--
-- Important:
-- This version requires the runtime to provide LuaSocket via require("socket").
-- If the module is unavailable in your controller image, the script will log
-- an explicit error and stop before trying to communicate.

G_HMI_CONNECTED = false
G_HMI_LAST_RX = nil

local HMI_IP = "192.168.192.25"
local HMI_PORT = 9000
local HMI_CONNECT_TIMEOUT_SEC = 3
local HMI_READ_TIMEOUT_SEC = 1
local HMI_KEEP_OPEN = true

local g_hmi_socket = nil

local function log(msg)
    textmsg("[socket-hmi] " .. tostring(msg))
end

local function load_socket_module()
    local ok, socket_or_err = pcall(require, "socket")
    if not ok then
        log("require(\"socket\") failed: " .. tostring(socket_or_err))
        return nil
    end

    return socket_or_err
end

local function connect_hmi_socket()
    local socket = load_socket_module()
    if not socket then
        return false
    end

    local client, create_err = socket.tcp()
    if not client then
        log("socket.tcp() failed: " .. tostring(create_err))
        return false
    end

    client:settimeout(HMI_CONNECT_TIMEOUT_SEC)

    local ok, connect_err = client:connect(HMI_IP, HMI_PORT)
    if ok ~= 1 and ok ~= true then
        log("connect failed: " .. tostring(connect_err))
        client:close()
        return false
    end

    client:settimeout(HMI_READ_TIMEOUT_SEC)
    g_hmi_socket = client
    G_HMI_CONNECTED = true
    log("connected to HMI " .. HMI_IP .. ":" .. tostring(HMI_PORT))
    return true
end

local function send_hmi_line(payload)
    if not g_hmi_socket then
        log("send skipped: socket is not connected")
        return false
    end

    local ok, send_err, bytes_sent = g_hmi_socket:send(tostring(payload) .. "\n")
    if not ok then
        log("send failed: " .. tostring(send_err) .. ", bytes_sent=" .. tostring(bytes_sent))
        return false
    end

    log("tx -> " .. tostring(payload))
    return true
end

local function recv_hmi_line()
    if not g_hmi_socket then
        log("receive skipped: socket is not connected")
        return nil
    end

    local line, recv_err, partial = g_hmi_socket:receive("*l")
    if line then
        G_HMI_LAST_RX = line
        log("rx <- " .. tostring(line))
        return line
    end

    if recv_err == "timeout" and partial and partial ~= "" then
        G_HMI_LAST_RX = partial
        log("rx partial <- " .. tostring(partial))
        return partial
    end

    log("receive failed: " .. tostring(recv_err))
    return nil
end

local function close_hmi_socket()
    if g_hmi_socket then
        local ok, close_err = pcall(function()
            g_hmi_socket:close()
        end)

        if not ok then
            log("close failed: " .. tostring(close_err))
        end
    end

    g_hmi_socket = nil
    G_HMI_CONNECTED = false
    log("socket closed")
end

local function run_demo_exchange()
    if not connect_hmi_socket() then
        log("HMI socket setup failed")
        return
    end

    send_hmi_line("ROBOT_HELLO")
    local rx = recv_hmi_line()

    if rx then
        send_hmi_line("ROBOT_ACK:" .. tostring(rx))
    end

    if not HMI_KEEP_OPEN then
        close_hmi_socket()
    end
end

run_demo_exchange()
