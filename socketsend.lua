local function log(msg)
    textmsg("[socket-send] " .. tostring(msg))
end

local hmi = rawget(_G, "HMI_SOCKET")
if not hmi then
    log("HMI socket object not found")
    return
end

local WAIT_REPLY_TIMEOUT_SEC = 120
local DEFAULT_READ_TIMEOUT_SEC = 2

-- Replace this string later with your real business payload.
local tx = "WAITING_FOR_BUSINESS_PAYLOAD"
local ok = hmi.ensure_connected()
if not ok then
    log("connect failed")
    return
end

if not hmi.send(tx) then
    log("send failed")
    return
end

log("tx sent, waiting up to " .. tostring(WAIT_REPLY_TIMEOUT_SEC) .. "s for HMI reply")
hmi.client:settimeout(WAIT_REPLY_TIMEOUT_SEC)
local rx = hmi.recv_line()
hmi.client:settimeout(DEFAULT_READ_TIMEOUT_SEC)

if rx then
    log("rx=" .. tostring(rx))
else
    log("no reply")
end
