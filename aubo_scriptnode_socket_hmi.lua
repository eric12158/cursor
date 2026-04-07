-- This is only a minimal wrapper for the current script runtime.
-- Do not expect another independent ScriptNode to reuse this live socket.
local hmi = require("aubo_hmi_socket")

local ok = hmi.open({
    ip = "192.168.192.25",
    port = 9000,
    connect_timeout = 3,
    read_timeout = 1,
    connected_flag = "ROBOT_CONNECTED",
})

if ok then
    textmsg("[socket-hmi-init] connected flag sent")
else
    textmsg("[socket-hmi-init] connect failed")
end
