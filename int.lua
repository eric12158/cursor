local host = "192.168.192.25"
local port = 9000
local connected_flag = "client successfully connect!"

_G.LOG = rawget(_G, "LOG") or function(tag, msg)
    if msg == nil then
        msg = tag
        tag = "LOG"
    end
    textmsg("[" .. tostring(tag) .. "] " .. tostring(msg))
end
local LOG = _G.LOG

local socket = assert(require("socket"))
local client = assert(socket.connect(host, port))

client:settimeout(0)
_G.HMI_SOCKET = { client = client }
client:send(connected_flag)
LOG("int", "client connected to HMI " .. host .. ":" .. tostring(port))
