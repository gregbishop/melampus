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
local LrPathUtils = import 'LrPathUtils'
local LrShell = import 'LrShell'

local Log = {}

--- Whether the folder has been consulted this module load, and whether it
-- was there (made if it had to be) when it was. The LrFileUtils calls that
-- check and make the folder yield, and a line logged inside a catalog write
-- gate (keywordFromPath in MelampusImport.lua warns from one) must not
-- yield; every run's first line is outside any gate, so the folder is
-- consulted at most once, by that line, whatever the outcome: a folder that
-- could not be made stays not made for the run rather than being asked for
-- again from inside a gate. Log.reveal(), which only the dialog calls and
-- never a gate, consults it again.
local folderChecked = false
local folderMade = false

--- Check the folder, making it when it is missing; folderMade says whether
-- it is there afterwards. The folder is checked, not io.open's result,
-- because io.open of a path whose folder is missing does not fail the same
-- way everywhere.
local function checkFolder()
	local folder = Log.folder()
	if not LrFileUtils.exists(folder) then
		LrFileUtils.createAllDirectories(folder)
	end
	folderMade = LrFileUtils.exists(folder) and true or false
	folderChecked = true
end

--- The log, open for appending, its folder made on first use; nil when the
-- folder could not be made or the file cannot be opened. A log that cannot
-- be written is not an error worth raising over in the middle of a run.
-- After the first line, io.open is the only call this makes.
local function open()
	if not folderChecked then checkFolder() end
	if not folderMade then return nil end
	return io.open(Log.path(), 'a')
end

--- One message is one line, whatever it holds. Messages carry text that is
-- not the plugin's (keyword names from the results file, file names), and
-- a line break in it would let that text start a timestamped line of its
-- own. Control characters are written as their escapes, \n \r \t and \xHH
-- for the rest, so the evidence stays readable in the log and never leaves
-- the entry it belongs to.
local ESCAPES = { ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function escape(text)
	return (string.gsub(text, '%c', function(c)
		return ESCAPES[c] or string.format('\\x%02X', string.byte(c))
	end))
end

local function write(level, message)
	local handle = open()
	if not handle then return end
	handle:write(os.date('%Y-%m-%d %H:%M:%S'), ' ', level, ' ', escape(tostring(message)), '\n')
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
	return LrPathUtils.child(Log.dataRoot(), 'logs')
end

--- Where the log file lives, for showing the user and for every message
-- that names it: <data root>/logs/Melampus.log, the one rule on both
-- platforms (card #442).
function Log.path()
	return LrPathUtils.child(Log.folder(), 'Melampus.log')
end

--- Show the log where it is: the file, selected in the folder that holds it
-- (LrShell reveals a file in its folder on both platforms). Made first,
-- with its folder, when nothing has been logged yet, so there is always
-- something to reveal; a folder that does not exist was what the old
-- button opened on Windows. The folder is consulted again here, never from
-- a gate, so one removed mid-session comes back from the button.
function Log.reveal()
	checkFolder()
	local handle = open()
	if handle then handle:close() end
	LrShell.revealInShell(Log.path())
end

return Log
