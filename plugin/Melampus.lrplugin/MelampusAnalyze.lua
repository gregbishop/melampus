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
local LrPathUtils = import 'LrPathUtils'
local LrTasks = import 'LrTasks'

local Log = require 'MelampusLog'

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

--- Where the CLI's own output goes. Not the null device: the cloud-primary
-- cost estimate (and any refusal, e.g. the model.max_images ceiling) prints to
-- stderr, and a non-interactive caller that discards it has erased the only
-- record of what a run was going to cost. Lives in the OS temp directory: the
-- shell does not create directories for a redirect, and temp is the one
-- location guaranteed to exist on both platforms (the plugin-log directory is
-- not — Log.path() is a macOS layout). Outside previewFolder so cleanUp()
-- does not take the evidence with it.
local function cliLogPath()
	return LrPathUtils.child(LrPathUtils.getStandardFilePath('temp'), 'melampus-cli.log')
end

--- Run the identification pipeline over a folder of previews, writing the
-- enriched results (quality and its rank, burst agreement, range flag,
-- encounter) to `resultsPath` in the same run.
-- Returns true plus the results path, or false plus a message.
function Analyze.run(previewFolder, resultsPath, profile)
	local folder = pluginDir()
	local executable = Analyze.executablePath()
	if not executable or not LrFileUtils.exists(executable) then
		return false, 'Melampus could not find its analysis program.\n\n'
			.. 'The plugin folder should contain a file named '
			.. Analyze.executableName() .. ':\n' .. tostring(folder)
			.. '\n\nCopy it there from the Melampus download and try again.'
	end

	local cliLog = cliLogPath()
	if WIN_ENV then
		-- cmd.exe expands %NAME% even inside double quotes, silently rewriting
		-- the path before execution. Refusing loudly beats running against a
		-- path the user never named. The message names the path that has the
		-- "%" and the fix for that path: the plugin folder is where the user
		-- put it; the other three are in the Windows temp folder.
		local inTemp = 'Melampus keeps this in the Windows temp folder. Set TEMP to a '
			.. 'folder whose path has no "%" and try again.'
		local checked = {
			{ 'plugin folder', folder,
				'Move the plugin to a folder whose path has no "%" and try again.' },
			{ 'previews folder', previewFolder, inTemp },
			{ 'results file', resultsPath, inTemp },
			{ 'log file', cliLog, inTemp },
		}
		for _, entry in ipairs(checked) do
			local what, path, advice = entry[1], tostring(entry[2]), entry[3]
			if path:find('%%') then
				return false, 'The ' .. what .. ' path contains "%", which the Windows '
					.. 'shell rewrites:\n' .. path .. '\n\n' .. advice
			end
		end
	end

	-- Identification and enrichment, one process. Long-running, so it must not
	-- be inside any write gate.
	-- --yes: this is a non-interactive caller, so the cloud-primary cost gate
	-- cannot ask. Selecting the photos and configuring a cloud backend with a
	-- key were the deliberate acts; the estimate is written to melampus-cli.log,
	-- and the model.max_images ceiling still refuses an oversized run outright.
	local command = shellLine(table.concat({
		quote(executable), quote(previewFolder),
		'--profile', quote(profile or 'wildlife'),
		'--plugin-out', quote(resultsPath),
		'--yes',
		'>' .. quote(cliLog) .. ' 2>&1',
	}, ' '))
	Log.info('running: ' .. command)
	local code = LrTasks.execute(command)
	if code ~= 0 then
		return false, 'Identification failed (exit ' .. tostring(code)
			.. ').\n\nSee the logs:\n' .. Log.path() .. '\n' .. cliLog
	end

	return true, resultsPath
end

return Analyze
