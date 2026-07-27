--[[ File logging.

     "It didn't seem to do anything" is unfalsifiable without a trace, so every
     action writes one. LrLogger's logfile target lands in
     ~/Documents/LrClassicLogs/Melampus.log on macOS. ]]
local LrLogger = import 'LrLogger'

local logger = LrLogger('Melampus')
logger:enable('logfile')

local Log = {}

local function stamp()
	return os.date('%H:%M:%S')
end

function Log.info(message)
	logger:info(stamp() .. ' ' .. tostring(message))
end

function Log.warn(message)
	logger:warn(stamp() .. ' ' .. tostring(message))
end

function Log.error(message)
	logger:error(stamp() .. ' ' .. tostring(message))
end

--- Where the log file lives, for showing the user.
-- Verified on macOS with Lightroom Classic 15.4: ~/Library/Logs/Adobe/Lightroom/
-- LrClassicLogs/Melampus.log. An earlier guess of ~/Documents/LrClassicLogs was
-- wrong and sent the user to a folder that did not exist.
function Log.path()
	local LrPathUtils = import 'LrPathUtils'
	local home = LrPathUtils.getStandardFilePath('home')
	return LrPathUtils.child(LrPathUtils.child(LrPathUtils.child(
		LrPathUtils.child(LrPathUtils.child(home, 'Library'), 'Logs'),
		'Adobe'), 'Lightroom'), 'LrClassicLogs/Melampus.log')
end

return Log
