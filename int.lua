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
_G.HMI_SOCKET = {
    client = client,
    clear = function()
        while true do
            local data, err, partial = client:receive()
            if data and data ~= "" then
                LOG("int", "clear rx <- " .. tostring(data))
            elseif partial and partial ~= "" then
                LOG("int", "clear partial <- " .. tostring(partial))
            else
                if err and err ~= "timeout" then
                    LOG("int", "clear buffer stop: " .. tostring(err))
                end
                break
            end
        end
    end
}
client:send(connected_flag)
LOG("int", "client connected to HMI " .. host .. ":" .. tostring(port))
