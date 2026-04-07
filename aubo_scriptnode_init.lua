-- AUBO ARCS/AuboStudio ScriptNode example
-- Purpose:
-- 1) Add one Modbus TCP signal for the HMI.
-- 2) Raise TCP by 100 mm when current target Z is lower than 200 mm.

G_HMI_CONNECTED = false

local HMI_IP = "192.168.192.25"
local HMI_PORT = 502
local HMI_SLAVE_ID = 1
local HMI_SIGNAL_NAME = "HMI_HR0"
local HMI_SIGNAL_TYPE = 3 -- 1=coil, 2=input status, 3=holding register, 4=input register
local HMI_SIGNAL_ADDR = 0
local CLEAR_ALL_MODBUS_SIGNALS = true

local function log(msg)
    textmsg("[script-node] " .. tostring(msg))
end

local function connect_hmi()
    local endpoint = HMI_IP .. "," .. tostring(HMI_PORT)
    local ok, err = pcall(function()
        if CLEAR_ALL_MODBUS_SIGNALS then
            modbusDeleteAllSignals()
        end

        modbusAddSignal(
            endpoint,
            HMI_SLAVE_ID,
            HMI_SIGNAL_TYPE,
            HMI_SIGNAL_ADDR,
            HMI_SIGNAL_NAME,
            0
        )
    end)

    G_HMI_CONNECTED = ok

    if ok then
        log("HMI signal added: " .. endpoint .. " -> " .. HMI_SIGNAL_NAME)
    else
        log("HMI connection setup failed: " .. tostring(err))
    end
end

local function raise_tcp_if_needed()
    local curr_pose = getTargetTcpPose()
    local z_threshold = 0.2
    local z_up = 0.1
    local curr_z = curr_pose[3]

    if curr_z < z_threshold then
        local target_pose = poseAdd(curr_pose, {0, 0, z_up, 0, 0, 0})

        setTcpOffset({0, 0, 0, 0, 0, 0})
        setSpeedFraction(0.1)
        moveLine(target_pose, 0.25, 0.25, 0.0, 0)

        log("TCP Z below threshold, raised by " .. tostring(z_up) .. " m")
    else
        log("TCP Z is safe, no lift needed")
    end
end

connect_hmi()
raise_tcp_if_needed()
log("Initialization complete")
