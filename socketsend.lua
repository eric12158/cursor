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

-- Replace this string later with your real business payload.
local tx = "WAITING_FOR_BUSINESS_PAYLOAD"
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
