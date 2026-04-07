local LOG = rawget(_G, "LOG") or function(msg)
    textmsg(tostring(msg))
end

local function fail(msg)
    LOG("[socket-send] " .. tostring(msg))
    error(msg)
end

local hmi = rawget(_G, "HMI_SOCKET")
if not hmi or not hmi.client then
    fail("HMI socket object not found")
end

local WAIT_REPLY_TIMEOUT_SEC = 120

-- Replace this string later with your real business payload.
local tx = "WAITING_FOR_BUSINESS_PAYLOAD"
hmi.clear_buffer()
LOG("[socket-send] buffer cleared")
local ok, send_err = hmi.client:send(tx .. "\n")
if not ok then
    fail("send failed: " .. tostring(send_err))
end

LOG("[socket-send] tx=" .. tostring(tx))
LOG("[socket-send] waiting up to " .. tostring(WAIT_REPLY_TIMEOUT_SEC) .. "s for HMI reply")
hmi.client:settimeout(WAIT_REPLY_TIMEOUT_SEC)
local rx, recv_err, partial = hmi.client:receive()
hmi.client:settimeout(0)

if rx then
    LOG("[socket-send] rx=" .. tostring(rx))
elseif partial and partial ~= "" then
    LOG("[socket-send] rx partial=" .. tostring(partial))
else
    fail("no reply within " .. tostring(WAIT_REPLY_TIMEOUT_SEC) .. "s")
end
