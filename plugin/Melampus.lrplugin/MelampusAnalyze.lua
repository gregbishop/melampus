--[[
Analyse photos that have no identification yet.

Until now the plugin could only apply results the Python side had already
computed, which meant selecting a folder it had never seen did nothing at all —
exactly the "science project" failure CLAUDE.md §5.4 warns about. This closes
that: the plugin exports previews itself, runs the pipeline, and comes back with
answers.

Two ordering constraints, both from §5.3 and both load-bearing:

  * requestJpegThumbnail is asynchronous. Every preview must be fully written
    before anything else happens, so the export loop waits on a completion count
    rather than assuming.
  * No write transaction is open during any of this. Exporting and running the
    model both yield; yielding inside a write gate is what produces "yielding is
    not allowed".
--]]

local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrPasswords = import 'LrPasswords'
local LrPathUtils = import 'LrPathUtils'
local LrTasks = import 'LrTasks'

local Json = require 'MelampusJson'
local Log = require 'MelampusLog'
local Rules = require 'MelampusRules'

local Analyze = {}

--- The executable ships inside the plugin folder, so the plugin never looks for
-- a Python environment. `melampus` on macOS, `melampus.exe` on Windows.
function Analyze.executableName()
	return WIN_ENV and 'melampus.exe' or 'melampus'
end

--- The plugin's own folder, or nil outside Lightroom.
local function pluginDir()
	return _PLUGIN and _PLUGIN.path
end

--- Absolute path of the executable beside this plugin, or nil outside Lightroom.
function Analyze.executablePath()
	local folder = pluginDir()
	if not folder then return nil end
	return LrPathUtils.child(folder, Analyze.executableName())
end

--- Write JPEG previews for the given photos into `folder`.
-- Returns how many were written. Blocks until every callback has fired, because
-- the pipeline cannot run against half-written files.
function Analyze.exportPreviews(photos, folder, longEdge, progress)
	LrFileUtils.createAllDirectories(folder)

	local requested, finished, written = 0, 0, 0
	local keepAlive = {}

	for index, photo in ipairs(photos) do
		if progress and progress:isCanceled() then break end
		local name = photo:getFormattedMetadata('fileName') or ('photo' .. index)
		local stem = string.match(name, '^(.+)%.[^.]+$') or name
		local target = LrPathUtils.child(folder, stem .. '.jpg')

		if LrFileUtils.exists(target) then
			-- Already exported on a previous attempt; do not pay for it twice.
			written = written + 1
		else
			requested = requested + 1
			-- The returned object must be retained or it is collected mid-flight
			-- and the callback never fires.
			keepAlive[#keepAlive + 1] = photo:requestJpegThumbnail(longEdge, longEdge,
				function(jpegData, errorString)
					if jpegData then
						local handle = io.open(target, 'wb')
						if handle then
							handle:write(jpegData)
							handle:close()
							written = written + 1
						else
							Log.warn('could not write preview for ' .. name)
						end
					else
						Log.warn('preview failed for ' .. name .. ': ' .. tostring(errorString))
					end
					finished = finished + 1
				end)
		end

		if progress then
			progress:setPortionComplete(index, #photos * 3)
		end
	end

	-- Wait for every outstanding callback. Yielding here is safe and required.
	local waited = 0
	while finished < requested and waited < 600 do
		LrTasks.sleep(0.2)
		waited = waited + 0.2
		if progress and progress:isCanceled() then break end
	end
	if finished < requested then
		Log.warn(string.format('%d previews did not complete within the timeout',
			requested - finished))
	end

	Log.info(string.format('previews written: %d of %d photos', written, #photos))
	return written
end

--- Total size of a folder of previews, in megabytes.
function Analyze.folderSizeMB(folder)
	local bytes = 0
	for file in LrFileUtils.files(folder) do
		local attrs = LrFileUtils.fileAttributes(file)
		bytes = bytes + ((attrs and attrs.fileSize) or 0)
	end
	return bytes / 1048576
end

--- Remove the working previews. They are a means, not an artefact worth keeping:
-- at roughly 78 KB each an unattended library sweep would leave a few hundred
-- megabytes lying in a temp folder that nothing ever cleans.
function Analyze.cleanUp(folder)
	if not folder or not LrFileUtils.exists(folder) then return 0 end
	local size = Analyze.folderSizeMB(folder)
	local ok = LrFileUtils.delete(folder)
	Log.info(string.format('removed preview folder (%.0f MB): %s', size, tostring(ok)))
	return size
end

--[[
Platform seams. Lightroom sets WIN_ENV / MAC_ENV globals; everything the shell
sees differs between them, so the differences live here and nowhere else.
LrTasks.execute runs the line through cmd.exe on Windows (double-quote quoting)
and sh on macOS (single-quote quoting).
--]]

--- Shell-quote a path so spaces and quotes survive the trip.
local function quote(text)
	text = tostring(text)
	if WIN_ENV then
		-- cmd.exe quoting. A Windows filename cannot contain a double quote,
		-- so wrapping is sufficient.
		return '"' .. text .. '"'
	end
	return "'" .. string.gsub(text, "'", "'\\''") .. "'"
end

--- The whole line, as the platform's shell needs to receive it.
-- cmd.exe /c, given a line that starts with a quote and holds more than two,
-- strips the first and the last one (see `cmd /?`); wrapped in a pair of its
-- own, the line loses only those and the quotes around each path survive.
local function shellLine(command)
	if WIN_ENV then return '"' .. command .. '"' end
	return command
end

--- Why the command must not run on Windows, or nil when it may.
-- cmd.exe expands %NAME% even inside double quotes, silently rewriting the
-- path before execution. Refusing loudly beats running against a path the
-- user never named. `checked` lists { what, path, advice }: the message
-- names the path that has the "%" and the fix for that path, the plugin
-- folder being where the user put it (MOVE_PLUGIN) and the rest in the
-- Windows temp folder (IN_TEMP).
local MOVE_PLUGIN = 'Move the plugin to a folder whose path has no "%" and try again.'
local IN_TEMP = 'Melampus keeps this in the Windows temp folder. Set TEMP to a '
	.. 'folder whose path has no "%" and try again.'

local function windowsPathRefusal(checked)
	if not WIN_ENV then return nil end
	for _, entry in ipairs(checked) do
		local what, path, advice = entry[1], tostring(entry[2]), entry[3]
		if path:find('%%') then
			return 'The ' .. what .. ' path contains "%", which the Windows '
				.. 'shell rewrites:\n' .. path .. '\n\n' .. advice
		end
	end
	return nil
end

--- A variable set in the child's environment, ahead of the command. LrTasks
-- .execute takes one string and nothing else, so the shell sets it: `VAR=
-- 'value' command` for sh, `set "VAR=value" && command` for cmd.exe. That
-- string is the child's command line for the run's duration, which is why
-- the caller logs the line with the value replaced, never this one.
local function environmentPrefix(name, value)
	if WIN_ENV then return 'set "' .. name .. '=' .. value .. '" && ' end
	return name .. '=' .. quote(value) .. ' '
end

--- Why the key must not go to cmd.exe, or nil when it may. Inside
-- `set "VAR=value"` a double quote ends the quoted text and what follows is
-- command text to cmd.exe, %NAME% is expanded even inside quotes (see
-- windowsPathRefusal), a line feed ends the line so what follows it is not
-- the line the plugin built, and a carriage return is dropped; none of them
-- can be escaped on a cmd.exe command line. sh gets the key through
-- quote(), where nothing needs refusing. The message never shows the key.
local function windowsKeyRefusal(key)
	if not WIN_ENV then return nil end
	if string.find(key, '["%%\r\n]') then
		return 'The API key kept for this engine contains a character the Windows '
			.. 'shell rewrites (", % or a line break), so Melampus will not hand it '
			.. 'to its analysis program.\n\nOpen Settings and enter the key again.'
	end
	return nil
end

--- Where the CLI's own output goes. Not the null device: the cloud-primary
-- cost estimate (and any refusal, e.g. the model.max_images ceiling) prints to
-- stderr, and a non-interactive caller that discards it has erased the only
-- record of what a run was going to cost. Lives in the OS temp directory: the
-- shell does not create directories for a redirect, and temp is the one
-- location guaranteed to exist on both platforms (the plugin-log directory is
-- not — Log.path() is a macOS layout). Outside previewFolder so cleanUp()
-- does not take the evidence with it.
local function tempPath(name)
	return LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), name)
end

local function cliLogPath()
	return tempPath('melampus-cli.log')
end

--- The one message for an executable that is not beside the plugin: which
-- folder should hold it and what the file is called, and nothing about how
-- it might be built.
local function missingExecutable()
	return 'Melampus could not find its analysis program.\n\n'
		.. 'The plugin folder should contain a file named '
		.. Analyze.executableName() .. ':\n' .. tostring(pluginDir())
		.. '\n\nCopy it there from the Melampus download and try again.'
end

--- The suffix every failure message that has a run behind it ends with: the
-- plugin's log and the CLI log of that run, so the two paths are said once.
local function seeTheLogs(cliLog)
	return '\n\nSee the logs:\n' .. Log.path() .. '\n' .. cliLog
end

--- Run the executable for one JSON answer: `flag` with stdout to `output`
-- under temp and stderr to the CLI log, decoded. Returns the value, or nil
-- plus a message: the executable is missing, exited non-zero, or printed
-- something `accept` does not recognise. `what` names the question for the
-- messages, `file` the output file for the Windows "%" refusal. Runs the
-- executable once per call.
local function askJson(flag, output, what, file, accept)
	local executable = Analyze.executablePath()
	if not executable or not LrFileUtils.exists(executable) then
		return nil, missingExecutable()
	end
	local target, cliLog = tempPath(output), cliLogPath()
	local refusal = windowsPathRefusal({
		{ 'plugin folder', pluginDir(), MOVE_PLUGIN },
		{ file, target, IN_TEMP },
		{ 'log file', cliLog, IN_TEMP },
	})
	if refusal then return nil, refusal end
	local command = shellLine(quote(executable) .. ' ' .. flag .. ' >'
		.. quote(target) .. ' 2>' .. quote(cliLog))
	Log.info('running: ' .. command)
	local code = LrTasks.execute(command)
	if code ~= 0 then
		return nil, 'Melampus could not ask its analysis program ' .. what
			.. ' (exit ' .. tostring(code) .. ').' .. seeTheLogs(cliLog)
	end
	local value, err = Json.decode(LrFileUtils.readFile(target) or '')
	if not accept(value) then
		return nil, 'Melampus did not understand what its analysis program said about ' .. what
			.. (err and (': ' .. tostring(err)) or '') .. '.\n\nWhat it printed is in:\n' .. target
			.. seeTheLogs(cliLog)
	end
	return value
end

--- Ask the executable which engines can run here: `--detect-engines` (card
-- #404) prints a JSON list of { engine, available, reason }. The Settings
-- dialog calls it once, when it opens.
function Analyze.detectEngines()
	return askJson('--detect-engines', 'melampus-engines.json', 'which engines can run here',
		'engines file', function(verdicts) return type(verdicts) == 'table' and verdicts[1] ~= nil end)
end

--- Ask the executable about the MLX model: `--model-status` (card #408)
-- prints one JSON object { repo, installed, bytes_total, bytes_done, path,
-- cancel_path }; bytes_total is null when the hub could not be reached. The
-- Settings dialog calls it once, when it opens.
function Analyze.modelStatus()
	return askJson('--model-status', 'melampus-model-status.json', 'the model',
		'model status file', function(status) return type(status) == 'table' and type(status.repo) == 'string' end)
end

--- Remove the MLX model from the cache: `--remove-model` (card #408), exit
-- 0 once it is gone. Returns true, or false plus a message with the CLI
-- log's tail (a download of it is running, or nothing is installed).
function Analyze.removeModel()
	local executable = Analyze.executablePath()
	if not executable or not LrFileUtils.exists(executable) then
		return false, missingExecutable()
	end
	local cliLog = cliLogPath()
	local command = shellLine(quote(executable) .. ' --remove-model >' .. quote(cliLog) .. ' 2>&1')
	Log.info('running: ' .. command)
	local code = LrTasks.execute(command)
	if code ~= 0 then
		return false, 'Melampus could not remove the model (exit ' .. tostring(code) .. ').\n\n'
			.. Analyze.tail(LrFileUtils.readFile(cliLog))
	end
	return true
end

--- The last lines of a log, for a message.
function Analyze.tail(text, lines)
	text = text or ''
	local kept, count = #text, 0
	for i = #text, 1, -1 do
		if string.sub(text, i, i) == '\n' then
			count = count + 1
			if count > (lines or 8) then break end
		end
		kept = i
	end
	return string.sub(text, kept)
end

-- ── the model download (card #408) ─────────────────────────────────────────
-- LrTasks.execute blocks and returns only the exit code: it cannot stream
-- stdout, and the SDK cannot kill the child. So the download's stdout (the
-- protocol lines, docs/config.md § Downloading the model) is redirected to
-- a file that a second task reads every second, and Cancel writes the
-- marker the executable watches, at the path --model-status named.

--- The download's stdout (the protocol lines) and stderr, under temp for
-- the reasons cliLogPath gives.
function Analyze.downloadFiles()
	return tempPath('melampus-download.progress'), tempPath('melampus-download.log')
end

--- The shell line that downloads the model, or nil plus the missing-
-- executable message.
function Analyze.downloadCommand()
	local executable = Analyze.executablePath()
	if not executable or not LrFileUtils.exists(executable) then
		return nil, missingExecutable()
	end
	local progress, log = Analyze.downloadFiles()
	return shellLine(quote(executable) .. ' --download-model >' .. quote(progress) .. ' 2>' .. quote(log))
end

--- Start the download. One task runs the command; another reads the
-- progress file every second and hands each update (Rules.parseDownloadLine)
-- to `onProgress`; when the command exits, `onFinish` gets the exit code
-- (0 done, 3 failed, 4 cancelled), the last update, and the tail of the
-- log. `cancelPath` is where --model-status said to write to cancel.
-- Returns a handle whose cancel() writes it, or nil plus a message. The
-- executable removes a stale marker when it starts, and its start (the
-- one-file unpack, the imports) takes seconds after the click, so a
-- cancel is held: once asked for, the poller writes the marker again on
-- every tick until the command exits, and a start-up removal loses it
-- for a second at most.
function Analyze.downloadModel(cancelPath, onProgress, onFinish)
	local command, err = Analyze.downloadCommand()
	if not command then return nil, err end
	local progressFile, logFile = Analyze.downloadFiles()
	-- Start clean: the shell truncates the file when the command starts,
	-- but the poller may read before that, and a previous run's `done`
	-- must not read as this run's.
	for _, path in ipairs({ progressFile, logFile }) do
		local handle = io.open(path, 'w')
		if handle then handle:close() end
	end
	local cancelled = false
	local function writeMarker()
		LrFileUtils.createAllDirectories(LrPathUtils.parent(cancelPath))
		local handle = io.open(cancelPath, 'w')
		if handle then handle:close() else Log.warn('could not write ' .. cancelPath) end
	end
	Log.info('running: ' .. command)
	local code = nil
	LrTasks.startAsyncTask(function() code = LrTasks.execute(command) end)
	LrTasks.startAsyncTask(function()
		while code == nil do
			LrTasks.sleep(1)
			-- Not after the exit: the executable removed the marker then,
			-- and one left behind would be the next run's stale one.
			if cancelled and code == nil then writeMarker() end
			local update = Rules.latestDownloadUpdate(LrFileUtils.readFile(progressFile))
			if update then onProgress(update) end
		end
		local update = Rules.latestDownloadUpdate(LrFileUtils.readFile(progressFile))
		Log.info('download exit ' .. tostring(code) .. ': ' .. tostring(update and update.state))
		onFinish(code, update, Analyze.tail(LrFileUtils.readFile(logFile)))
	end)
	return {
		cancel = function()
			cancelled = true
			writeMarker()
		end,
	}
end

--- Run the identification pipeline over a folder of previews, writing the
-- enriched results (quality and its rank, burst agreement, range flag,
-- encounter) to `resultsPath` in the same run. `engine` is the engine
-- preference (Rules.ENGINES); nil or empty leaves the choice to the CLI.
-- Returns true plus the results path, or false plus a message.
function Analyze.run(previewFolder, resultsPath, profile, engine)
	local folder = pluginDir()
	local executable = Analyze.executablePath()
	if not executable or not LrFileUtils.exists(executable) then
		return false, missingExecutable()
	end

	local chosen, engineError = Rules.chosenEngine({ engine = engine })
	if engineError then return false, engineError end

	local cliLog = cliLogPath()
	local refusal = windowsPathRefusal({
		{ 'plugin folder', folder, MOVE_PLUGIN },
		{ 'previews folder', previewFolder, IN_TEMP },
		{ 'results file', resultsPath, IN_TEMP },
		{ 'log file', cliLog, IN_TEMP },
	})
	if refusal then return false, refusal end

	-- Identification and enrichment, one process. Long-running, so it must not
	-- be inside any write gate.
	-- --yes: this is a non-interactive caller, so the cloud-primary cost gate
	-- cannot ask. Selecting the photos and configuring a cloud backend with a
	-- key were the deliberate acts; the estimate is written to melampus-cli.log,
	-- and the model.max_images ceiling still refuses an oversized run outright.
	local parts = {
		quote(executable), quote(previewFolder),
		'--profile', quote(profile or 'wildlife'),
	}
	-- --backend only when the user chose an engine; otherwise the CLI decides.
	if chosen then
		parts[#parts + 1] = '--backend'
		parts[#parts + 1] = quote(chosen)
	end
	parts[#parts + 1] = '--plugin-out'
	parts[#parts + 1] = quote(resultsPath)
	parts[#parts + 1] = '--yes'
	parts[#parts + 1] = '>' .. quote(cliLog) .. ' 2>&1'
	local line = table.concat(parts, ' ')

	-- A cloud engine's key (card #405): stored by the Settings dialog through
	-- LrPasswords, handed to the executable in the variable it reads, and
	-- only for the engine the user picked. It is never an argument and never
	-- logged; the log carries the line with the key blanked.
	local logged = line
	local variable = Rules.keyVariable(chosen)
	local key = variable and LrPasswords.retrieve(variable)
	if key and key ~= '' then
		local keyRefusal = windowsKeyRefusal(key)
		if keyRefusal then return false, keyRefusal end
		line = environmentPrefix(variable, key) .. line
		logged = environmentPrefix(variable, '') .. logged
	end
	local command = shellLine(line)
	Log.info('running: ' .. shellLine(logged))
	local code = LrTasks.execute(command)
	if code ~= 0 then
		return false, 'Identification failed (exit ' .. tostring(code) .. ').' .. seeTheLogs(cliLog)
	end

	return true, resultsPath
end

return Analyze
