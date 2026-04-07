local hmi = require("aubo_hmi_socket")

local ok = hmi.open({
    ip = "192.168.192.25",
    port = 9000,
    connect_timeout = 3,
    read_timeout = 2,
})

if not ok then
    textmsg("[hmi-coords] connect failed")
    return
end

local coords = hmi.request_coords("GET_COORDS")
if not coords then
    textmsg("[hmi-coords] invalid coordinate reply")
    hmi.close()
    return
end

textmsg(
    string.format(
        "[hmi-coords] x=%.3f y=%.3f z=%.3f rx=%.3f ry=%.3f rz=%.3f",
        coords[1], coords[2], coords[3], coords[4], coords[5], coords[6]
    )
)

-- Example:
-- moveLine(coords, 0.25, 0.25, 0.0, 0)

hmi.close()
