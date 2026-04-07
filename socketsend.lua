local HMI_IP = "192.168.192.25"
local HMI_PORT = 9000
local HMI_CONNECT_TIMEOUT_SEC = 3
local HMI_READ_TIMEOUT_SEC = 2

local function log(msg)
    textmsg("[socket-send] " .. tostring(msg))
end

local function build_fallback_socket()
    local ok, socket_or_err = pcall(require, "socket")
    if not ok then
        log("require(\"socket\") failed: " .. tostring(socket_or_err))
        return nil
    end

    local client, create_err = socket_or_err.tcp()
    if not client then
        log("socket.tcp() failed: " .. tostring(create_err))
        return nil
    end

    client:settimeout(HMI_CONNECT_TIMEOUT_SEC)

    local connected, connect_err = client:connect(HMI_IP, HMI_PORT)
    if connected ~= 1 and connected ~= true then
        log("connect failed: " .. tostring(connect_err))
        client:close()
        return nil
    end

    client:settimeout(HMI_READ_TIMEOUT_SEC)

    local wrapper = {}

    function wrapper.send_and_wait(payload)
        local send_ok, send_err, bytes_sent = client:send(tostring(payload) .. "\n")
        if not send_ok then
            log("send failed: " .. tostring(send_err) .. ", bytes_sent=" .. tostring(bytes_sent))
            return nil
        end

        local line, recv_err, partial = client:receive("*l")
        if line then
            return line
        end

        if recv_err == "timeout" and partial and partial ~= "" then
            return partial
        end

        if recv_err ~= "timeout" then
            log("receive failed: " .. tostring(recv_err))
        end

        return nil
    end

    return wrapper
end

local hmi = rawget(_G, "HMI_SOCKET")
if not hmi then
    log("global HMI socket not found, using fallback connection")
    hmi = build_fallback_socket()
end

if not hmi then
    log("connect failed")
    return
end

-- Replace this string later with your real business payload.
local tx = "WAITING_FOR_BUSINESS_PAYLOAD"
local rx = hmi.send_and_wait(tx)

if rx then
    log("rx=" .. tostring(rx))
else
    log("no reply")
end
