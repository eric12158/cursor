local M = {}

local socket_lib = nil
local client = nil
local active_cfg = nil

local function log(msg)
    textmsg("[hmi-socket] " .. tostring(msg))
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
    return socket_lib
end

local function normalize_cfg(cfg)
    cfg = cfg or {}
    return {
        ip = cfg.ip or "192.168.192.25",
        port = cfg.port or 9000,
        connect_timeout = cfg.connect_timeout or 3,
        read_timeout = cfg.read_timeout or 1,
        connected_flag = cfg.connected_flag,
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

    active_cfg = normalize_cfg(cfg)

    local tcp_client, create_err = socket.tcp()
    if not tcp_client then
        log("socket.tcp() failed: " .. tostring(create_err))
        return false
    end

    tcp_client:settimeout(active_cfg.connect_timeout)

    local ok, connect_err = tcp_client:connect(active_cfg.ip, active_cfg.port)
    if ok ~= 1 and ok ~= true then
        log("connect failed: " .. tostring(connect_err))
        tcp_client:close()
        return false
    end

    tcp_client:settimeout(active_cfg.read_timeout)
    client = tcp_client
    log("connected to HMI " .. active_cfg.ip .. ":" .. tostring(active_cfg.port))

    if active_cfg.connected_flag then
        M.send(active_cfg.connected_flag)
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

function M.request_line(payload)
    if not M.send(payload) then
        return nil
    end

    return M.recv_line()
end

function M.parse_coords_csv(line)
    if not line then
        return nil
    end

    local values = {}
    for token in string.gmatch(line, "[^,]+") do
        local number = tonumber(token)
        if not number then
            return nil
        end
        values[#values + 1] = number
    end

    if #values ~= 6 then
        return nil
    end

    return values
end

function M.request_coords(cmd)
    local line = M.request_line(cmd or "GET_COORDS")
    return M.parse_coords_csv(line)
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
    active_cfg = nil
    log("socket closed")
end

return M
