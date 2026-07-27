--[[
Import identifications into the catalog.

The ordering here is the whole point, and it follows CLAUDE.md §5.3: every piece
of reading, parsing and decision-making finishes *before* a write transaction
opens. Yielding inside withWriteAccessDo is what produces "yielding is not
allowed" errors, and file I/O yields.

So: read the file, decode it, read existing photo state, build every plan — then
open one transaction per chunk and apply. Nothing async happens inside a gate.

Dry run is the default. The first run against any catalog reports what it would
do and writes nothing.
--]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'
local LrPrefs = import 'LrPrefs'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local Json = require 'MelampusJson'
local Log = require 'MelampusLog'
local Rules = require 'MelampusRules'

local PLUGIN_ID = 'net.gregbishop.melampus'
local CHUNK = 100 -- photos per write transaction, so a crash loses little

--- Look for the results file in the obvious places before giving up.
-- The plugin lives at <repo>/plugin/Melampus.lrplugin, so the file the Python
-- side writes is normally two levels up. Finding it automatically removes the
-- single configuration step that made the plugin look broken.
local function discoverResults()
	local candidates = {}
	local pluginDir = _PLUGIN and _PLUGIN.path
	if pluginDir then
		local repo = LrPathUtils.parent(LrPathUtils.parent(pluginDir))
		if repo then
			candidates[#candidates + 1] = LrPathUtils.child(repo, 'plugin_results.json')
			candidates[#candidates + 1] = LrPathUtils.child(repo, 'stage1_full_results.json')
		end
		candidates[#candidates + 1] = LrPathUtils.child(pluginDir, 'plugin_results.json')
	end
	for _, candidate in ipairs(candidates) do
		if LrFileUtils.exists(candidate) then
			Log.info('auto-discovered results at ' .. candidate)
			return candidate
		end
	end
	return nil
end

local function readResults(path)
	if not path or path == '' then
		path = discoverResults()
	end
	if not path or path == '' then
		return nil, 'FIRST_RUN'
	end
	if not LrFileUtils.exists(path) then
		return nil, 'Results file not found:\n' .. path
	end
	local text = LrFileUtils.readFile(path)
	if not text or text == '' then
		return nil, 'Results file is empty:\n' .. path
	end
	local data, err = Json.decode(text)
	if not data then
		return nil, 'Could not parse the results file.\n\n' .. tostring(err)
	end
	if type(data) ~= 'table' then
		return nil, 'Results file did not contain a list of records.'
	end
	return data, nil, path
end

--- Flatten one Python-side record into the shape Rules expects.
local function normalise(record)
	local ident = record.identification
	local top, alternates = nil, {}
	if type(ident) == 'table' and type(ident.candidates) == 'table' then
		for index, cand in ipairs(ident.candidates) do
			if index == 1 then
				top = cand
			else
				alternates[#alternates + 1] = cand.common_name
			end
		end
	end
	local abstain = false
	if type(ident) == 'table' and ident.abstain == true then abstain = true end

	return {
		file = record.file,
		status = record.status,
		species = top and top.common_name or nil,
		scientificName = top and top.scientific_name or nil,
		confidence = top and top.confidence or nil,
		alternates = #alternates > 0 and table.concat(alternates, ', ') or nil,
		taxon = type(ident) == 'table' and ident.taxon or nil,
		abstain = abstain,
		-- These three are supplied by the encounter-aware exporter. Absent is
		-- handled: Rules treats a missing agreement as "unknown", not "failed".
		burstAgreement = record.burst_agreement,
		rangeFlag = record.range_flag == true,
		encounter = record.encounter,
		quality = record.quality,
		model = record.model,
	}
end

--- Existing catalog state for a photo, as a plain table Rules can reason about.
local function photoState(photo)
	local keywords = {}
	for _, keyword in ipairs(photo:getRawMetadata('keywords') or {}) do
		local parts, node = {}, keyword
		while node do
			table.insert(parts, 1, node:getName())
			node = node:getParent()
		end
		keywords[#keywords + 1] = table.concat(parts, ' > ')
	end

	local stored = {}
	for _, field in ipairs({ 'species', 'model', 'schemaVersion', 'verdict' }) do
		stored[field] = photo:getPropertyForPlugin(_PLUGIN, field)
	end

	return {
		rating = photo:getRawMetadata('rating'),
		pickStatus = photo:getRawMetadata('pickStatus'),
		colorNameForLabel = photo:getRawMetadata('colorNameForLabel'),
		keywords = keywords,
		melampus = stored,
	}
end

--- Find or create a keyword from a "A > B > C" path. Must run inside a write gate.
local function keywordFromPath(catalog, path)
	local parent = nil
	for segment in string.gmatch(path, '[^>]+') do
		local name = string.gsub(segment, '^%s*(.-)%s*$', '%1')
		if name ~= '' then
			-- returnExisting = true, so re-running never duplicates the tree.
			parent = catalog:createKeyword(name, {}, false, parent, true)
		end
	end
	return parent
end

local function describe(plan)
	local bits = {}
	if plan.rating then bits[#bits + 1] = 'rating ' .. plan.rating end
	if plan.label then bits[#bits + 1] = 'label ' .. plan.label end
	if plan.pickStatus then bits[#bits + 1] = 'flag ' .. plan.pickStatus end
	if #plan.keywords > 0 then bits[#bits + 1] = #plan.keywords .. ' keywords' end
	local fields = 0
	for _ in pairs(plan.metadata or {}) do fields = fields + 1 end
	if fields > 0 then bits[#bits + 1] = fields .. ' fields' end
	return table.concat(bits, ', ')
end

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusImport', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local catalog = LrApplication.activeCatalog()

		local settings = Rules.defaultSettings()
		for key in pairs(settings) do
			if prefs[key] ~= nil then settings[key] = prefs[key] end
		end

		Log.info('=== import started ===')
		Log.info('results path: ' .. tostring(prefs.resultsPath))
		Log.info('dryRun=' .. tostring(settings.dryRun)
			.. ' writeKeywords=' .. tostring(settings.writeKeywords)
			.. ' writeMetadata=' .. tostring(settings.writeMetadata)
			.. ' force=' .. tostring(settings.force))

		local records, err, resolvedPath = readResults(prefs.resultsPath)
		if resolvedPath and resolvedPath ~= prefs.resultsPath then
			-- Remember what we found so Settings shows it and the next run is direct.
			prefs.resultsPath = resolvedPath
		end
		if not records then
			Log.error('could not read results: ' .. tostring(err))
			if err == 'FIRST_RUN' then
				-- §5.4.6: a first run should explain itself, not error.
				LrDialogs.message('Welcome to Melampus',
					'Melampus works out what species are in your photos, on your own Mac, '
					.. 'and then adds them to your photos as keywords.\n\n'
					.. 'It has not been told where those identifications are yet.\n\n'
					.. 'What to do:\n'
					.. '1.  Go to Library > Plug-in Extras > Melampus: Settings…\n'
					.. '2.  Under Step 1, click Choose… and pick the file\n'
					.. '     plugin_results.json in your melampus folder.\n'
					.. '3.  Leave "Preview only" ticked.\n'
					.. '4.  Come back here and run Identify Selected Photos again.\n\n'
					.. 'Nothing will be changed on your photos until you untick '
					.. '"Preview only" yourself.', 'info')
			else
				LrDialogs.message('Melampus', err, 'critical')
			end
			return
		end
		Log.info('records loaded: ' .. tostring(#records))

		local byFile = {}
		for _, record in ipairs(records) do
			if type(record) == 'table' and record.file then
				byFile[record.file] = record
				-- Results come from exported JPEGs; the catalog holds raws.
				-- Match on basename so 0A1A2475.jpg finds 0A1A2475.CR3.
				local stem = string.match(record.file, '^(.+)%.[^.]+$')
				if stem then byFile[stem] = record end
			end
		end

		local photos = catalog:getTargetPhotos()
		Log.info('photos selected: ' .. tostring(#photos))
		if #photos == 0 then
			Log.warn('nothing selected')
			LrDialogs.message('Melampus', 'Select some photos first.', 'info')
			return
		end

		local progress = LrProgressScope({
			title = settings.dryRun and 'Melampus: previewing changes'
				or 'Melampus: writing metadata',
			functionContext = context,
		})
		progress:setCancelable(true)

		-- ── phase 1: read and decide. No writes, yielding is fine here. ──────
		local planned, matched, unmatched = {}, 0, 0
		for index, photo in ipairs(photos) do
			if progress:isCanceled() then break end
			progress:setPortionComplete(index - 1, #photos * 2)

			local name = photo:getFormattedMetadata('fileName') or ''
			local stem = string.match(name, '^(.+)%.[^.]+$') or name
			local record = byFile[name] or byFile[stem]

			if record then
				matched = matched + 1
				local plan = Rules.planFor(normalise(record), photoState(photo), settings)
				if not Rules.isEmpty(plan) then
					planned[#planned + 1] = { photo = photo, plan = plan, name = name }
				end
			else
				unmatched = unmatched + 1
				if unmatched <= 5 then
					Log.warn('no result for: ' .. name .. '  (looked for "' .. stem .. '")')
				end
			end
		end
		Log.info(string.format('matched %d, unmatched %d, with changes %d',
			matched, unmatched, #planned))

		-- ── dry run stops here ──────────────────────────────────────────────
		if not Rules.shouldApply(settings) then
			local lines = {
				'PREVIEW ONLY — none of your photos were changed.\n',
				string.format('You selected %d photos.', #photos),
				string.format('Melampus has identifications for %d of them.', matched),
			}
			if unmatched > 0 then
				lines[#lines + 1] = string.format(
					'%d had no identification and were left alone.', unmatched)
			end
			lines[#lines + 1] = string.format('\n%d photos would get new keywords:\n', #planned)
			for i = 1, math.min(#planned, 15) do
				lines[#lines + 1] = '  ' .. planned[i].name .. ': ' .. describe(planned[i].plan)
			end
			if #planned > 15 then
				lines[#lines + 1] = string.format('  … and %d more.', #planned - 15)
			end
			if #planned == 0 and matched > 0 then
				lines[#lines + 1] = 'Nothing to change — these photos already have their '
					.. 'Melampus keywords. That is what a second run should do.'
			end
			lines[#lines + 1] = '\nHappy with this? Go to Melampus: Settings… and untick '
				.. '"Preview only", then run this again to apply it.'
			progress:done()
			Log.info('dry run complete; nothing written')
			lines[#lines + 1] = '\nLog: ' .. Log.path()
			LrDialogs.message('Melampus — dry run', table.concat(lines, '\n'), 'info')
			return
		end

		-- ── phase 2: apply. Chunked, and nothing async inside the gate. ─────
		local written, chunkStart = 0, 1
		while chunkStart <= #planned do
			if progress:isCanceled() then break end
			local chunkStop = math.min(chunkStart + CHUNK - 1, #planned)

			catalog:withWriteAccessDo('Melampus: apply identifications', function()
				for i = chunkStart, chunkStop do
					local entry = planned[i]
					local photo, plan = entry.photo, entry.plan

					if plan.rating ~= nil then photo:setRawMetadata('rating', plan.rating) end
					if plan.label ~= nil then photo:setRawMetadata('colorNameForLabel', plan.label) end
					if plan.pickStatus ~= nil then photo:setRawMetadata('pickStatus', plan.pickStatus) end

					for _, path in ipairs(plan.keywords) do
						local keyword = keywordFromPath(catalog, path)
						if keyword then photo:addKeyword(keyword) end
					end

					for field, value in pairs(plan.metadata or {}) do
						photo:setPropertyForPlugin(_PLUGIN, field, tostring(value))
					end
					written = written + 1
				end
			end, { timeout = 60 })

			progress:setPortionComplete(#photos + chunkStop, #photos * 2)
			chunkStart = chunkStop + 1
		end

		-- Timestamp separately: it is plugin-private, so it stays off the undo
		-- stack rather than cluttering it with a bookkeeping entry.
		catalog:withPrivateWriteAccessDo(function()
			local stamp = os.date('%Y-%m-%dT%H:%M:%S')
			for i = 1, math.min(written, #planned) do
				planned[i].photo:setPropertyForPlugin(_PLUGIN, 'processedAt', stamp)
			end
		end)

		Log.info('photos updated: ' .. tostring(written))
		progress:done()
		local suffix = progress:isCanceled()
			and '\n\nCancelled — completed work was kept.' or ''
		LrDialogs.message('Melampus — done',
			string.format('Added keywords to %d of your %d selected photos.\n\n'
				.. '%d had no identification and were left untouched.\n\n'
				.. 'Next: run "Melampus: Set Up Review Collections" to get a '
				.. 'Needs Review collection you can work through.%s',
				written, #photos, unmatched, suffix), 'info')
	end)
end)
