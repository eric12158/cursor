local hmi = require("int")

local ok = hmi.ensure_connected()
if not ok then
    textmsg("[socket-send] connect failed")
    return
end

-- Replace this string later with your real business payload.
local tx = "WAITING_FOR_BUSINESS_PAYLOAD"
local rx = hmi.send_and_wait(tx)

if rx then
    textmsg("[socket-send] rx=" .. tostring(rx))
else
    textmsg("[socket-send] no reply")
end
