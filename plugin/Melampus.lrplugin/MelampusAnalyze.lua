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

--- Where the repository lives, inferred from the plugin's own location.
function Analyze.repoRoot()
	local pluginDir = _PLUGIN and _PLUGIN.path
	if not pluginDir then return nil end
	return LrPathUtils.parent(LrPathUtils.parent(pluginDir))
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

--- Shell-quote a path so spaces and quotes survive the trip.
local function quote(text)
	return "'" .. string.gsub(tostring(text), "'", "'\\''") .. "'"
end

--- Run the identification pipeline over a folder of previews.
-- Returns true plus the results path, or false plus a message.
function Analyze.run(repo, previewFolder, resultsPath)
	local python = LrPathUtils.child(LrPathUtils.child(repo, '.venv'), 'bin/python')
	if not LrFileUtils.exists(python) then
		return false, 'Could not find the Melampus Python environment at:\n' .. python
			.. '\n\nRun this once in Terminal, from the melampus folder:\n'
			.. '  uv venv --python 3.12 .venv\n'
			.. '  uv pip install --python .venv/bin/python -e "./service[dev]"'
	end

	local melampus = LrPathUtils.child(LrPathUtils.child(repo, '.venv'), 'bin/melampus-id')
	local raw = LrPathUtils.child(previewFolder, '_raw_results.json')

	-- Identification. Long-running, so it must not be inside any write gate.
	local command = table.concat({
		'cd', quote(repo), '&&',
		quote(melampus), quote(previewFolder),
		'--json-out', quote(raw),
		'>/dev/null 2>&1',
	}, ' ')
	Log.info('running: ' .. command)
	local code = LrTasks.execute(command)
	if code ~= 0 then
		return false, 'Identification failed (exit ' .. tostring(code)
			.. ').\n\nSee the log:\n' .. Log.path()
	end

	-- Enrich with burst agreement, range flags and quality so the write gates
	-- and star ratings have something to work with.
	local enrich = table.concat({
		'cd', quote(repo), '&&',
		quote(LrPathUtils.child(LrPathUtils.child(repo, '.venv'), 'bin/python')),
		quote(LrPathUtils.child(LrPathUtils.child(repo, 'tools'), 'make_plugin_results.py')),
		quote(previewFolder), quote(raw), quote(resultsPath),
		'--occurrence', '--quality',
		'>/dev/null 2>&1',
	}, ' ')
	Log.info('running: ' .. enrich)
	code = LrTasks.execute(enrich)
	if code ~= 0 then
		return false, 'Post-processing failed (exit ' .. tostring(code)
			.. ').\n\nSee the log:\n' .. Log.path()
	end

	return true, resultsPath
end

return Analyze
