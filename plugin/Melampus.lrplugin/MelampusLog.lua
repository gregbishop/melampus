--[[ File logging.

     "It didn't seem to do anything" is unfalsifiable without a trace, so every
     action writes one. The plugin writes the file itself, at Log.path(), one
     line per message (card #442). It did use LrLogger's 'logfile' target,
     which writes wherever Lightroom's native AgFileLogger decides:
     ~/Library/Logs/Adobe/Lightroom/LrClassicLogs on macOS, and on Windows a
     folder the SDK neither names nor lets a plugin set (LrLogger:enable takes
     a target name or a table of functions, never a path), so the path the
     Settings dialog showed was a guess, and wrong on Windows. ]]
local LrFileUtils = import 'LrFileUtils'

local Log = {}

--- The log, open for appending, its folder made on first use; nil when the
-- folder cannot be made or the file cannot be opened. A log that cannot be
-- written is not an error worth raising over in the middle of a run. The
-- folder is checked, not the call's result, because io.open of a path whose
-- folder is missing does not fail the same way everywhere.
local function open()
	local folder = Log.folder()
	if not LrFileUtils.exists(folder) then
		LrFileUtils.createAllDirectories(folder)
		if not LrFileUtils.exists(folder) then return nil end
	end
	return io.open(Log.path(), 'a')
end

local function write(level, message)
	local handle = open()
	if not handle then return end
	handle:write(os.date('%Y-%m-%d %H:%M:%S'), ' ', level, ' ', tostring(message), '\n')
	handle:close()
end

function Log.info(message)
	write('INFO', message)
end

function Log.warn(message)
	write('WARN', message)
end

function Log.error(message)
	write('ERROR', message)
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
