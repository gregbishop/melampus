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

--- The per-user Melampus data directory: the root the executable keeps its
-- config and caches under (service/melampus/config.py, _data_root), so the
-- log sits beside them. ~/Library/Application Support/Melampus on macOS,
-- %LOCALAPPDATA%\Melampus on Windows. The plugin cannot read %LOCALAPPDATA%
-- (the SDK's sandbox keeps only os.clock, os.date, os.time and os.tmpname of
-- `os`), so it builds the variable's default, <home>\AppData\Local, which is
-- the executable's own fallback when the variable is unset. The plugin
-- needs this before it can run anything, since it logs the first command,
-- so it cannot ask the executable the way it asks for cancel_path.
function Log.dataRoot()
	local LrPathUtils = import 'LrPathUtils'
	local home = LrPathUtils.getStandardFilePath('home')
	local base
	if WIN_ENV then
		base = LrPathUtils.child(LrPathUtils.child(home, 'AppData'), 'Local')
	else
		base = LrPathUtils.child(LrPathUtils.child(home, 'Library'), 'Application Support')
	end
	return LrPathUtils.child(base, 'Melampus')
end

--- The folder that holds the log: <data root>/logs.
function Log.folder()
	return import('LrPathUtils').child(Log.dataRoot(), 'logs')
end

--- Where the log file lives, for showing the user and for every message
-- that names it: <data root>/logs/Melampus.log, the one rule on both
-- platforms (card #442).
function Log.path()
	return import('LrPathUtils').child(Log.folder(), 'Melampus.log')
end

return Log
